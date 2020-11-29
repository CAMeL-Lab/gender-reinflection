import torch
import torch.nn.functional as F
import re

class BatchSampler:
    def __init__(self, model, src_vocab_char,
                 src_vocab_word, trg_vocab_char,
                 src_labels_vocab, trg_labels_vocab,
                 trg_gender_vocab):

        self.model = model
        self.src_vocab_char = src_vocab_char
        self.src_vocab_word = src_vocab_word
        self.trg_vocab_char = trg_vocab_char
        self.src_labels_vocab = src_labels_vocab
        self.trg_labels_vocab = trg_labels_vocab
        self.trg_gender_vocab = trg_gender_vocab

    def set_batch(self, batch):
        self.sample_batch = batch

    def get_trg_sentence(self, index):
        trg_sentence = self.sample_batch['trg_y'][index].cpu().detach().numpy()
        return self.get_str_sentence(trg_sentence, self.trg_vocab_char)

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
        trg_gender = self.sample_batch['trg_gender'][index].cpu().detach().numpy().tolist()
        return self.trg_gender_vocab.lookup_index(trg_gender)

    def get_trg_label(self, index):
        trg_label = self.sample_batch['trg_label'][index].cpu().detach().numpy().tolist()
        return self.trg_labels_vocab.lookup_index(trg_label)

    def get_src_label(self, index):
        src_label = self.sample_batch['src_label'][index].cpu().detach().numpy().tolist()
        return self.src_labels_vocab.lookup_index(src_label)

    def greedy_decode(self, sentence, trg_gender=None, max_len=512):
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
        if trg_gender:
            vectorized_trg_gender = self.trg_gender_vocab.lookup_token(trg_gender)
            vectorized_trg_gender = torch.tensor([vectorized_trg_gender], dtype=torch.long)
        else:
            vectorized_trg_gender = None

        # converting the lists to tensors
        vectorized_src_sentence_char = torch.tensor([vectorized_src_sentence_char], dtype=torch.long)
        vectorized_src_sentence_word = torch.tensor([vectorized_src_sentence_word], dtype=torch.long)
        src_sentence_length = torch.tensor(src_sentence_length, dtype=torch.long)

        # passing the src sentence to the encoder
        with torch.no_grad():
            encoder_outputs, encoder_h_t = self.model.encoder(vectorized_src_sentence_char,
                                                              vectorized_src_sentence_word,
                                                              src_sentence_length
                                                              )

        # creating attention mask
        attention_mask = self.model.create_mask(vectorized_src_sentence_char, self.src_vocab_char.pad_idx)

        # initializing the first decoder_h_t to encoder_h_t
        decoder_h_t = encoder_h_t

        context_vectors = torch.zeros(1, self.model.encoder.rnn.hidden_size * 2)

        # intializing the trg sequences to the <s> token
        trg_seqs = [self.trg_vocab_char.sos_idx]

        with torch.no_grad():
            for i in range(max_len):
                y_t = torch.tensor([trg_seqs[-1]], dtype=torch.long)

                # do a single decoder step
                prediction, decoder_h_t, atten_scores, context_vectors = self.model.decoder(trg_seqs=y_t,
                                                                                            encoder_outputs=encoder_outputs,
                                                                                            decoder_h_t=decoder_h_t,
                                                                                            context_vectors=context_vectors,
                                                                                            attention_mask=attention_mask,
                                                                                            trg_gender=vectorized_trg_gender
                                                                                            )

                # getting the most probable prediciton
                max_pred = torch.argmax(prediction, dim=1).item()

                # if we reach </s> token, stop decoding
                if max_pred == self.trg_vocab_char.eos_idx:
                    break

                trg_seqs.append(max_pred)

        str_sentence = self.get_str_sentence(trg_seqs, self.trg_vocab_char)
        return str_sentence
