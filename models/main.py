import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from data_utils import SeqVocabulary, RawDataset
import json
import random
import numpy as np
import argparse
from seq2seq import Seq2Seq
from nmt_sampler import NMT_Batch_Sampler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Vectorizer:
    """Vectorizer Class"""
    def __init__(self, src_vocab, trg_vocab):
        """src_vocab and trg_vocab are on the char
        level"""
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    @classmethod
    def create_vectorizer(cls, data_examples):
        """Class method which builds the vectorizer
        vocab"""

        src_vocab = SeqVocabulary()
        trg_vocab = SeqVocabulary()

        for ex in data_examples:
            src = ex.src
            trg = ex.trg

            for t in src:
                src_vocab.add_token(t)

            for t in trg:
                trg_vocab.add_token(t)

        return cls(src_vocab, trg_vocab)

    def get_src_indices(self, seq):
        """
        Args:
          - seq (str): The src sequence

        Returns:
          - indices (list): <s> + List of tokens to index mapping + </s>
        """
        indices = [self.src_vocab.sos_idx]
        indices.extend([self.src_vocab.lookup_token(t) for t in seq])
        indices.append(self.src_vocab.eos_idx)
        return indices

    def get_trg_indices(self, seq):
        """
        Args:
          - seq (str): The trg sequence

        Returns:
          - trg_x_indices (list): <s> + List of tokens to index mapping
          - trg_y_indices (list): List of tokens to index mapping + </s>
        """
        indices = [self.trg_vocab.lookup_token(t) for t in seq]

        trg_x_indices = [self.trg_vocab.sos_idx] + indices
        trg_y_indices = indices + [self.trg_vocab.eos_idx]
        return trg_x_indices, trg_y_indices

    def vectorize(self, src, trg):
        """
        Args:
          - src (str): The src sequence
          - src (str): The trg sequence
        Returns:
          - vectorized_src
          - vectorized_trg_x
          - vectorized_trg_y
        """
        src = src
        trg = trg

        vectorized_src = self.get_src_indices(src)
        vectorized_trg_x, vectorized_trg_y = self.get_trg_indices(trg)

        return {'src': torch.tensor(vectorized_src, dtype=torch.long),
                'trg_x': torch.tensor(vectorized_trg_x, dtype=torch.long),
                'trg_y': torch.tensor(vectorized_trg_y, dtype=torch.long)
               }

    def to_serializable(self):
        return {'src_vocab': self.src_vocab.to_serializable(),
                'trg_vocab': self.trg_vocab.to_serializable()
               }

    @classmethod
    def from_serializable(cls, contents):
        src_vocab = SeqVocabulary.from_serializable(contents['src_vocab'])
        trg_vocab = SeqVocabulary.from_serializable(contents['trg_vocab'])
        return cls(src_vocab, trg_vocab)


class MT_Dataset(Dataset):
    """MT Dataset as a PyTorch dataset"""
    def __init__(self, raw_dataset, vectorizer):
        self.vectorizer = vectorizer
        self.train_examples = raw_dataset.train_examples
        self.dev_examples = raw_dataset.dev_examples
        self.test_examples = raw_dataset.test_examples
        self.lookup_split = {'train': self.train_examples,
                             'dev': self.dev_examples,
                             'test': self.test_examples}
        self.set_split('train')

    def get_vectorizer(self):
        return self.vectorizer

    @classmethod
    def load_data_and_create_vectorizer(cls, data_dir):
        raw_dataset = RawDataset(data_dir)
        # Note: we always create the vectorized based on the train examples
        vectorizer = Vectorizer.create_vectorizer(raw_dataset.train_examples)
        return cls(raw_dataset, vectorizer)

    @classmethod
    def load_data_and_load_vectorizer(cls, data_dir, vec_path):
        raw_dataset = RawDataset(data_dir)
        vectorizer = cls.load_vectorizer(vec_path)
        return cls(raw_dataset, vectorizer)

    @staticmethod
    def load_vectorizer(vec_path):
        with open(vec_path) as f:
            return Vectorizer.from_serializable(json.load(f))

    def save_vectorizer(self, vec_path):
        with open(vec_path, 'w') as f:
            return json.dump(self.vectorizer.to_serializable(), f)

    def set_split(self, split):
        self.split = split
        self.split_examples = self.lookup_split[self.split]
        return self.split_examples

    def __getitem__(self, index):
        example = self.split_examples[index]
        src, trg = example.src, example.trg
        vectorized = self.vectorizer.vectorize(src, trg)
        return vectorized

    def __len__(self):
        return len(self.split_examples)

class Collator:
    def __init__(self, src_pad_idx, trg_pad_idx):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def __call__(self, batch):
        # Sorting the batch by src seqs length in descending order
        sorted_batch = sorted(batch, key=lambda x: x['src'].shape[0], reverse=True)

        src_seqs = [x['src'] for x in sorted_batch]
        trg_x_seqs = [x['trg_x'] for x in sorted_batch]
        trg_y_seqs = [x['trg_y'] for x in sorted_batch]
        lengths = [len(seq) for seq in src_seqs]

        padded_src_seqs = pad_sequence(src_seqs, batch_first=True, padding_value=self.src_pad_idx)
        padded_trg_x_seqs = pad_sequence(trg_x_seqs, batch_first=True, padding_value=self.trg_pad_idx)
        padded_trg_y_seqs = pad_sequence(trg_y_seqs, batch_first=True, padding_value=self.trg_pad_idx)
        lengths = torch.tensor(lengths, dtype=torch.long)

        return {'src': padded_src_seqs,
                'trg_x': padded_trg_x_seqs,
                'trg_y': padded_trg_y_seqs,
                'src_lengths': lengths}

def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def train(model, dataloader, optimizer, criterion, device='cpu', teacher_forcing_prob=1):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        src = batch['src']
        trg_x = batch['trg_x']
        trg_y = batch['trg_y']
        src_lengths = batch['src_lengths']

        preds = model(src_seqs=src,
                      src_seqs_lengths=src_lengths,
                      trg_seqs=trg_x,
                      teacher_forcing_prob=teacher_forcing_prob)

        # CrossEntropysLoss accepts matrices always! 
        # the preds must be of size (N, C) where C is the number 
        # of classes and N is the number of samples. 
        # The ground truth must be a Vector of size C!
        preds = preds.contiguous().view(-1, preds.shape[-1])
        trg_y = trg_y.view(-1)

        loss = criterion(preds, trg_y)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device='cpu', teacher_forcing_prob=0):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            src = batch['src']
            trg_x = batch['trg_x']
            trg_y = batch['trg_y']
            src_lengths = batch['src_lengths']

            preds = model(src_seqs=src,
                          src_seqs_lengths=src_lengths,
                          trg_seqs=trg_x,
                          teacher_forcing_prob=teacher_forcing_prob)
            # CrossEntropyLoss accepts matrices always! 
            # the preds must be of size (N, C) where C is the number 
            # of classes and N is the number of samples. 
            # The ground truth must be a Vector of size C!
            preds = preds.contiguous().view(-1, preds.shape[-1])
            trg_y = trg_y.view(-1)

            loss = criterion(preds, trg_y)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def inference(sampler, dataloader, preds_dir):
    output_file = open(preds_dir, mode='w', encoding='utf8')

    for batch in dataloader:
        sampler.update_batch(batch)
        src = sampler.get_src_sentence(0)
        trg = sampler.get_trg_sentence(0)
        pred = sampler.get_pred_sentence(0)
        translated = sampler.translate_sentence(src)

        output_file.write(translated)
        output_file.write('\n')
        logger.info(f'src: ' + src)
        logger.info(f'trg: ' + trg)
        logger.info(f'pred: ' + pred)
        logger.info(f'trans: ' + translated)
        # print(src)
        # print(trg)
        # print(pred)
        # print(translated)
        # print(len(translated))
        # train_log.write(f'src: ' + src)
        # train_log.write('\n')
        # train_log.write(f'trg: ' + trg)
        # train_log.write('\n')
        # train_log.write(f'pred: ' + pred)
        # train_log.write('\n')
        # train_log.write(f'trans: ' + translated)
        # train_log.write('\n\n')
        # train_preds.write(pred)
        # train_preds.write('\n')
        # train_preds_inf.write(translated)
        # train_preds_inf.write('\n')
    # train_log.close()
    # train_preds.close()
    # train_preds_inf.close()
    output_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the src and trg files."
        )
    parser.add_argument(
        "--vectorizer_path",
        default=None,
        type=str,
        help="The path of the saved vectorizer"
    )
    parser.add_argument(
        "--cache_files",
        action="store_true",
        help="Whether to cache the vocab and the vectorizer objects or not"
        )
    parser.add_argument(
        "--reload_files",
        action="store_true",
        help="Whether to reload the vocab and the vectorizer objects from a cached file"
        )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform."
        )
    parser.add_argument(
        "--embedding_dim",
        default=32,
        type=int,
        help="The embedding dimensions of the model"
    )
    parser.add_argument(
        "--hidd_dim",
        default=64,
        type=int,
        help="The hidden dimensions of the model"
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-4,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU"
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Whether to use the gpu or not."
        )
    parser.add_argument(
        "--seed",
        default=21,
        type=int,
        help="Random seed."
        )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        default=None,
        help="The directory of the model."
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training or not."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval or not."
    )
    parser.add_argument(
        "--do_inference",
        action="store_true",
        help="Whether to do inference or not."
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="dev",
        help="The dataset to do inference on."
    )
    parser.add_argument(
        "--preds_dir",
        type=str,
        default=None,
        help="The directory to write the translations to"
    )
    args = parser.parse_args()
    # args = argparse.Namespace(data_dir='/home/ba63/gender-bias/data/christine_2019/Arabic-parallel-gender-corpus',
    #                     vectorizer_path='/home/ba63/gender-bias/models/saved_models/char_level_vectorizer.json',
    #                     reload_files=False,
    #                     cache_files=False,
    #                     num_epochs=50,
    #                     embedding_dim=32,
    #                     hidd_dim=64,
    #                     learning_rate=5e-4,
    #                     use_cuda=True,
    #                     batch_size=64,
    #                     seed=21,
    #                     model_path='/home/ba63/gender-bias/models/saved_models/char_level_model_small_old.pt'
    #                     )

    device = torch.device('cuda' if args.use_cuda else 'cpu')
    set_seed(args.seed, args.use_cuda)

    if args.reload_files:
        dataset = MT_Dataset.load_data_and_load_vectorizer(args.data_dir, args.vectorizer_path)
    else:
        dataset = MT_Dataset.load_data_and_create_vectorizer(args.data_dir)

    if args.cache_files:
        dataset.save_vectorizer(args.vectorizer_path)

    vectorizer = dataset.get_vectorizer()
    ENCODER_INPUT_DIM = len(vectorizer.src_vocab)
    DECODER_INPUT_DIM = len(vectorizer.trg_vocab)
    DECODER_OUTPUT_DIM = len(vectorizer.trg_vocab)
    SRC_PAD_INDEX = vectorizer.src_vocab.pad_idx
    TRG_PAD_INDEX = vectorizer.trg_vocab.pad_idx

    model = Seq2Seq(encoder_input_dim=ENCODER_INPUT_DIM,
                    encoder_embed_dim=args.embedding_dim,
                    encoder_hidd_dim=args.hidd_dim,
                    decoder_input_dim=DECODER_INPUT_DIM,
                    decoder_embed_dim=args.embedding_dim,
                    decoder_output_dim=DECODER_OUTPUT_DIM,
                    src_padding_idx=SRC_PAD_INDEX,
                    trg_padding_idx=TRG_PAD_INDEX)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_INDEX)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    patience=2, factor=0.5)
    collator = Collator(SRC_PAD_INDEX, TRG_PAD_INDEX)
    model = model.to(device)

    if args.do_train:
        logger.info('Training...')
        train_losses = []
        dev_losses = []
        best_loss = 1e10
        teacher_forcing_prob = 0.3
        set_seed(args.seed, args.use_cuda)
        for epoch in range(args.num_train_epochs):
            dataset.set_split('train')
            dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collator)

            train_loss = train(model, dataloader, optimizer, criterion, device, teacher_forcing_prob=teacher_forcing_prob)
            train_losses.append(train_loss)

            dataset.set_split('dev')
            dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collator)
            dev_loss = evaluate(model, dataloader, criterion, device, teacher_forcing_prob=0)
            dev_losses.append(dev_loss)

            #save best model
            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save(model.state_dict(), args.model_path)

            scheduler.step(dev_loss)
            logger.info(f'Epoch: {(epoch + 1)}')
            logger.info(f'\tTrain Loss: {train_loss:.4f}   |   Dev Loss: {dev_loss:.4f}')

    if args.do_eval:
        logger.info('Evaluation')
        set_seed(args.seed, args.use_cuda)
        dev_losses = []
        for epoch in range(args.num_train_epochs):
            dataset.set_split('dev')
            dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collator)
            dev_loss = evaluate(model, dataloader, criterion, device, teacher_forcing_prob=0)
            dev_losses.append(dev_loss)
            logger.info(f'Dev Loss: {dev_loss:.4f}')

    if args.do_inference:
        logger.info('Inference')
        set_seed(args.seed, args.use_cuda)
        model.load_state_dict(torch.load(args.model_path))
        device = torch.device('cpu')
        model = model.to(device)
        dataset.set_split(args.inference_mode)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
        sampler = NMT_Batch_Sampler(model, vectorizer.src_vocab, vectorizer.trg_vocab)
        inference(sampler, dataloader, args.preds_dir)


if __name__ == "__main__":
    main()
