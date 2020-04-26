import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encoder biGRU"""
    def __init__(self, input_dim, embed_dim,
                 hidd_dim, padding_idx=0):

        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(input_dim, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(embed_dim, hidd_dim, batch_first=True, bidirectional=True)

    def forward(self, src_seqs, src_lengths):

        embedded_seqs = self.embedding_layer(src_seqs)

        packed_seqs = pack_padded_sequence(embedded_seqs, src_lengths, batch_first=True)

        output, h_t = self.rnn(packed_seqs)

        #output is a packed_sequence
        #h_t shape: [num_layers * num_dirs, batch_size, hidd_dim]

        #reshaping h_t to [batch_size, num_layers * num_dirs * hidd_dim]

        h_t = h_t.permute(1, 0, 2) #[batch_size, num_layers * num_dirs, hidd_dim]
        #Note: when we call permute, the contiguity of a tensor is lost,
        #so we have to call contiguous before reshaping!
        h_t = h_t.contiguous().view(h_t.shape[0], -1) #[batch_size, num_layers * num_dirs * hidd_dim]

        #unpacking output
        unpacked_output, lengths = pad_packed_sequence(output, batch_first=True)
        #output shape: [batch_size, src_seq_length, hidd_dim * num_dirs]

        return unpacked_output, h_t

class Decoder(nn.Module):
    """Decoder GRU

       Things to note:
           - The input to the decoder rnn at each time step is the
             concatenation of the embedded token and the context vector
           - The context vector will have a size of batch_size, hidd_dim
           - Note that the decoder hidd_dim == the encoder hidd_dim * 2
           - The prediction layer input is the concatenation of
             the context vector and the h_t of the decoder
    """

    def __init__(self, input_dim, embed_dim,
                 hidd_dim, output_dim,
                 attention,
                 padding_idx=0,
                 sos_idx=2,
                 eos_idx=3):

        super(Decoder, self).__init__()
        self.hidd_dim = hidd_dim
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.attention = attention
        self.embedding_layer = nn.Embedding(input_dim, embed_dim, padding_idx=padding_idx)
        # the input to the rnn is the context_vector + embedded token --> embed_dim + hidd_dim
        self.rnn = nn.GRUCell(embed_dim + hidd_dim, hidd_dim)
        # the input to the classifier is h_t + context_vector --> hidd_dim * 2
        self.classifier = nn.Linear(hidd_dim * 2, output_dim)

        # sampling temperature
        self.sampling_temperature = 3

    def forward(self, trg_seqs, encoder_outputs, encoder_h_t, mask,
                teacher_forcing_prob=0.3,
                inference=False,
                max_len=200):

        # if we're doing inference
        if trg_seqs is None:
            teacher_forcing_prob = 0
            trg_seq_len = max_len
            batch_size = encoder_outputs.shape[0]
            inference = True
        else:
            # reshaping trg_seqs to: [trg_seq_len, batch_size]
            trg_seqs = trg_seqs.permute(1, 0)
            trg_seq_len, batch_size = trg_seqs.shape

        # initializing the context_vectors to zeros
        context_vectors = torch.zeros(batch_size, self.hidd_dim)
        # moving the context_vectors to the right device
        context_vectors = context_vectors.to(encoder_h_t.device)

        # initializing the hidden state to the encoder hidden state
        h_t = encoder_h_t

        # initializing the first trg input to the <sos> token
        y_t = torch.ones(batch_size, dtype=torch.long) * self.sos_idx
        # moving the y_t to the right device
        y_t = y_t.to(encoder_h_t.device)

        outputs = []
        self.attention_scores = []

        for i in range(trg_seq_len):

            teacher_forcing = np.random.random() < teacher_forcing_prob

            # Step 1: Concat the embedded token and the context_vectors
            embedded = self.embedding_layer(y_t)

            rnn_input = torch.cat((embedded, context_vectors), dim=1)

            # Step 2: Do a single RNN step and update the decoder hidden state
            h_t = self.rnn(rnn_input, h_t)

            # Step 3: Calculate attention and update context vectors
            context_vectors, attention_probs = self.attention(encoder_outputs, h_t, mask)

            # Step 4: Obtain the predicion vector
            prediction_vector = torch.cat((h_t, context_vectors), dim=1)

            # Step 5: Obtain the prediction output
            pred_output = self.classifier(prediction_vector)

            # if teacher_forcing, use ground truth target tokens
            # as an input to the decoder in the next time_step
            if teacher_forcing:
                y_t = trg_seqs[i]

            # If not teacher_forcing force, use the maximum 
            # prediction as an input to the decoder in 
            # the next time step
            if not teacher_forcing:
                # we multiply the predictions with a sampling_temperature
                # to make the propablities peakier, so we can be confident about the
                # maximum prediction
                pred_output_probs = F.softmax(pred_output * self.sampling_temperature, dim=1)
                y_t = torch.argmax(pred_output_probs, dim=1)

                # if we predicted the <eos> token stop decoding
                if inference and (y_t == self.eos_idx).sum() == y_t.shape[0]:
#                     print('inference')
                    break

            outputs.append(pred_output)
            self.attention_scores.append(attention_probs)

        outputs = torch.stack(outputs).permute(1, 0, 2)
        # outputs shape: [batch_size, trg_seq_len, output_dim]
        return outputs


class AdditiveAttention(nn.Module):
    """Attention mechanism as a MLP
    as used by Bahdanau et. al 2015"""

    def __init__(self, encoder_hidd_dim, decoder_hidd_dim):
        super(AdditiveAttention, self).__init__()
        self.atten = nn.Linear((encoder_hidd_dim * 2) + decoder_hidd_dim, decoder_hidd_dim)
        self.v = nn.Linear(decoder_hidd_dim, 1)

    def forward(self, key_vectors, query_vector, mask):
        """key_vectors: encoder hidden states.
           query_vector: decoder hidden state at time t
           mask: the mask vector of zeros and ones
        """

        #key_vectors shape: [batch_size, src_seq_length, encoder_hidd_dim * 2]
        #query_vector shape: [batch_size, decoder_hidd_dim]
        #Note: encoder_hidd_dim * 2 == decoder_hidd_dim

        batch_size, src_seq_length, encoder_hidd_dim = key_vectors.shape

        #changing the shape of query_vector to [batch_size, src_seq_length, decoder_hidd_dim]
        #we will repeat the query_vector src_seq_length times at dim 1
        query_vector = query_vector.unsqueeze(1).repeat(1, src_seq_length, 1)

        # Step 1: Compute the attention scores through a MLP

        # concatenating the key_vectors and the query_vector
        atten_input = torch.cat((key_vectors, query_vector), dim=2)
        # atten_input shape: [batch_size, src_seq_length, (encoder_hidd_dim * 2) + decoder_hidd_dim]

        atten_scores = self.atten(atten_input)
        # atten_scores shape: [batch_size, src_seq_length, decoder_hidd_dim]

        atten_scores = torch.tanh(atten_scores)

        # mapping atten_scores from decoder_hidd_dim to 1
        atten_scores = self.v(atten_scores)

        # atten_scores shape: [batch_size, src_seq_length, 1]
        atten_scores = atten_scores.squeeze(dim=2)
        # atten_scores shape: [batch_size, src_seq_length]

        # masking the atten_scores
        atten_scores = atten_scores.masked_fill(mask == 0, -1e10)

        # Step 2: normalizing atten_scores through a softmax to get probs
        atten_scores = F.softmax(atten_scores, dim=1)

        # Step 3: computing the new context vector
        context_vectors = torch.matmul(key_vectors.permute(0, 2, 1), atten_scores.unsqueeze(2)).squeeze(dim=2)

        # context_vectors shape: [batch_size, encoder_hidd_dim * 2]

        return context_vectors, atten_scores

class Seq2Seq(nn.Module):
    def __init__(self, encoder_input_dim, encoder_embed_dim,
                 encoder_hidd_dim, decoder_input_dim, decoder_embed_dim,
                 decoder_output_dim, src_padding_idx=0, trg_padding_idx=0):

        super(Seq2Seq, self).__init__()
        self.src_padding_idx = src_padding_idx
        self.encoder = Encoder(input_dim=encoder_input_dim,
                               embed_dim=encoder_embed_dim,
                               hidd_dim=encoder_hidd_dim,
                               padding_idx=src_padding_idx)

        decoder_hidd_dim = encoder_hidd_dim * 2

        self.attention = AdditiveAttention(encoder_hidd_dim=encoder_hidd_dim,
                                           decoder_hidd_dim=decoder_hidd_dim)

        self.decoder = Decoder(input_dim=decoder_input_dim,
                               embed_dim=decoder_embed_dim,
                               hidd_dim=decoder_hidd_dim,
                               output_dim=decoder_output_dim,
                               attention=self.attention,
                               padding_idx=trg_padding_idx)

    def create_mask(self, src_seqs, src_padding_idx):
        mask = (src_seqs != src_padding_idx)
        # mask shape: [batch_size, src_seq_length]
        return mask

    def forward(self, src_seqs, src_seqs_lengths, trg_seqs, teacher_forcing_prob=0.3):
        encoder_output, encoder_h_t = self.encoder(src_seqs, src_seqs_lengths)
        mask = self.create_mask(src_seqs, self.src_padding_idx)
        decoder_output = self.decoder(trg_seqs, encoder_output, encoder_h_t, mask,
                                      teacher_forcing_prob=teacher_forcing_prob)
        return decoder_output
