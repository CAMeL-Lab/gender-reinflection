import torch

class NMT_Batch_Sampler:
    def __init__(self, model, src_vocab, trg_vocab):
        self.model = model
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def update_batch(self, batch):
        self.sample_batch = batch

        src = batch['src']
        trg_x = batch['trg_x']
        trg_y = batch['trg_y']
        src_lengths = batch['src_lengths']

        preds = self.model(src, src_lengths, trg_x, teacher_forcing_prob=0)
        # preds shape: [batch_size, trg_seq_len, output_dim]

        attention_scores = [score.cpu().detach() for score in self.model.decoder.attention_scores]
        # len(attention_scores): trg_seq_len
        # each vector in attention_scores has a shape: [batch_size, src_seq_len]
        attention_scores = torch.stack(attention_scores)
        attention_scores = attention_scores.permute(1, 0, 2)
        # attention_score shape: [batch_size, trg_seq_len, src_seq_len]

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
        src_sentence = self.sample_batch['src'][index].cpu().detach().numpy()
        return self.get_str_sentence(src_sentence, self.src_vocab)

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

    def translate_sentence(self, sentence):
        """Args:
            - sentence (string): sentence to be translated
           Returns:
            - translated sentence (string)
        """

        # vectorizing the sentence
        vectorized_src_sentence = [self.src_vocab.sos_idx]
        vectorized_src_sentence.extend([self.src_vocab.lookup_token(t) for t in sentence])
        vectorized_src_sentence.append(self.src_vocab.eos_idx)

        # getting sentence length
        src_sentence_length = len(vectorized_src_sentence)

        # converting the vectorized sentence and the length to tensors
        vectorized_src_sentence = torch.tensor([vectorized_src_sentence], dtype=torch.long)
        src_sentence_length = torch.tensor([src_sentence_length], dtype=torch.long)

        with torch.no_grad():
            decoder_output = self.model(src_seqs=vectorized_src_sentence,
                                        src_seqs_lengths=src_sentence_length,
                                        trg_seqs=None, teacher_forcing_prob=0)


        max_preds = torch.argmax(decoder_output, dim=2).squeeze().cpu().detach().tolist()

        max_preds = [max_preds] if isinstance(max_preds, int) else max_preds
        str_sentence = self.get_str_sentence(max_preds, self.trg_vocab)

        return str_sentence