import torch
import numpy as np
import re

class NMT_Batch_Sampler:
    def __init__(self, model, src_vocab_char,
                 src_vocab_word, trg_vocab,
                 src_gender_vocab, trg_gender_vocab):

        self.model = model
        self.src_vocab_char = src_vocab_char
        self.src_vocab_word = src_vocab_word
        self.trg_vocab = trg_vocab
        self.src_gender_vocab = src_gender_vocab
        self.trg_gender_vocab = trg_gender_vocab

    def update_batch(self, batch):
        self.sample_batch = batch

        src_char = batch['src_char']
        src_word = batch['src_word']
        trg_x = batch['trg_x']
        trg_y = batch['trg_y']
        src_lengths = batch['src_lengths']
        trg_gender = batch['trg_g']

       # turn off teacher forcing during inference
        preds, attention_scores = self.model(char_src_seqs=src_char,
                                             word_src_seqs=src_word,
                                             src_seqs_lengths=src_lengths,
                                             trg_seqs=trg_x,
                                             trg_gender=trg_gender,
                                             teacher_forcing_prob=0)

        # preds shape: [batch_size, trg_seq_len, output_dim]

        self.sample_batch['preds'] = preds
        self.sample_batch['attention_scores'] = attention_scores

    def get_pred_sentence(self, index):
        preds = self.sample_batch['preds']

        max_preds = torch.argmax(preds, dim=2)
        # max_preds shape: [batch_size, trg_seq_len]
        max_pred_sentence = max_preds[index].cpu().detach().numpy()
        return self.get_str_sentence(max_pred_sentence, self.trg_vocab)

    def get_trg_sentence(self, index):
        trg_sentence = self.sample_batch['trg_y'][index].cpu().detach().numpy()
        return self.get_str_sentence(trg_sentence, self.trg_vocab)

    def get_src_sentence(self, index):
        src_sentence = self.sample_batch['src_char'][index].cpu().detach().numpy()
        return self.get_str_sentence(src_sentence, self.src_vocab_char)

    def get_str_sentence(self, vectorized_sentence, vocab):
        sentence = []
        for i in vectorized_sentence:
            if i == vocab.sos_idx:
                continue
            elif i == vocab.eos_idx:
                break
            else:
                sentence.append(vocab.lookup_index(i))
        return ''.join(sentence)

    def get_trg_gender(self, index):
        trg_gender = self.sample_batch['trg_g'][index].cpu().detach().numpy().tolist()
        return self.trg_gender_vocab.lookup_index(trg_gender)

    def get_src_gender(self, index):
        src_gender = self.sample_batch['src_g'][index].cpu().detach().numpy().tolist()
        return self.src_gender_vocab.lookup_index(src_gender)

    def translate_sentence(self, sentence, trg_gender, max_len=512):
        # vectorizing the src sentence on the char level and word level

        sentence = re.split(r'(\s+)', sentence)
        vectorized_src_sentence_char = [self.src_vocab_char.sos_idx]
        vectorized_src_sentence_word = [self.src_vocab_word.sos_idx]

        for word in sentence:
            for c in word:
                vectorized_src_sentence_char.append(self.src_vocab_char.lookup_token(c))
                vectorized_src_sentence_word.append(self.src_vocab_word.lookup_token(word))

        vectorized_src_sentence_word.append(self.src_vocab_word.eos_idx)
        vectorized_src_sentence_char.append(self.src_vocab_char.eos_idx)

        # getting sentence length
        src_sentence_length = [len(vectorized_src_sentence_char)]

        # vectorizing the trg gender
        vectorized_trg_gender = self.trg_gender_vocab.lookup_token(trg_gender)

        # converting the lists to tensors
        vectorized_src_sentence_char = torch.tensor([vectorized_src_sentence_char], dtype=torch.long)
        vectorized_src_sentence_word = torch.tensor([vectorized_src_sentence_word], dtype=torch.long)
        src_sentence_length = torch.tensor(src_sentence_length, dtype=torch.long)
        vectorized_trg_gender = torch.tensor([vectorized_trg_gender], dtype=torch.long)

        # passing the src sentence to the encoder
        with torch.no_grad():
            encoder_outputs, encoder_h_t= self.model.encoder(vectorized_src_sentence_char,
                                                             vectorized_src_sentence_word,
                                                             src_sentence_length)

        # creating attention mask
        attention_mask = self.model.create_mask(vectorized_src_sentence_char, self.src_vocab_char.pad_idx)

        # initilizating the first decoder_h_t to encoder_h_t
        #decoder_h_t = encoder_h_t
        decoder_h_t = torch.tanh(self.model.linear_map(encoder_h_t))

        if self.model.decoder.rnn.num_layers > 1:
            decoder_h_t = decoder_h_t.expand(self.model.decoder.rnn.num_layers,
                                                decoder_h_t.shape[0],
                                                decoder_h_t.shape[1]
                                                ).contiguous()
            # decoder_h_t shape: [num_layers, batch_size, decoder_hidd_dim]
        else:
            decoder_h_t = decoder_h_t.unsqueeze(0)
            # decoder_h_t shape: [1, batch_size, decoder_hidd_dim]


        # initializing the context vectors to 0
        #context_vectors = torch.zeros(1, self.model.decoder.hidd_dim)

        context_vectors = torch.zeros(1, self.model.encoder.rnn.hidden_size * 2)

        # intializing the trg sequences to the <s> token
        trg_seqs = [self.trg_vocab.sos_idx]

        with torch.no_grad():
            for i in range(max_len):
                y_t = torch.tensor([trg_seqs[-1]], dtype=torch.long)

                # do a single decoder step
                prediction, decoder_h_t, atten_scores, context_vectors = self.model.decoder(trg_seqs=y_t,
                                                                                            trg_gender=vectorized_trg_gender,
                                                                                            encoder_outputs=encoder_outputs,
                                                                                            decoder_h_t=decoder_h_t,
                                                                                            context_vectors=context_vectors,
                                                                                            attention_mask=attention_mask)

                # getting the most probable prediciton
                max_pred = torch.argmax(prediction, dim=1).item()

                # if we reach </s> token, stop decoding
                if max_pred == self.trg_vocab.eos_idx:
                    break

                trg_seqs.append(max_pred)

        str_sentence = self.get_str_sentence(trg_seqs, self.trg_vocab)
        return str_sentence
