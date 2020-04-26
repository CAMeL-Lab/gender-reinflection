import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encoder bi-GRU"""
    def __init__(self, input_dim, embed_dim,
                 hidd_dim, padding_idx=0):

        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(input_dim, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(embed_dim, hidd_dim, batch_first=True, bidirectional=True)

    def forward(self, src_seqs, src_seqs_lengths):

        embedded_seqs = self.embedding_layer(src_seqs)
        # packing the embedded_seqs
        packed_embedded_seqs = pack_padded_sequence(embedded_seqs, src_seqs_lengths, batch_first=True)

        output, hidd = self.rnn(packed_embedded_seqs)
        # hidd shape: [num_layers * num_dirs, batch_size, hidd_dim]

        # changing hidd shape to: [batch_size, num_layers * num_dirs, hidd_dim]
        hidd = hidd.permute(1, 0 ,2)

        # changing hidd shape to: [batch_size, num_layers * num_dirs * hidd_dim]
        hidd = hidd.contiguous().view(hidd.shape[0], -1)

        # unpacking the output
        output, lengths = pad_packed_sequence(output, batch_first=True)
        # output shape: [batch_size, src_seqs_length, num_dirs * hidd_dim]
        return output, hidd


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
                 padding_idx=0):

        super(Decoder, self).__init__()
        self.hidd_dim = hidd_dim
        self.attention = attention
        self.embedding_layer = nn.Embedding(input_dim, embed_dim, padding_idx=padding_idx)
        # the input to the rnn is the context_vector + embedded token --> embed_dim + hidd_dim
        self.rnn = nn.GRUCell((embed_dim + hidd_dim), hidd_dim)
        # the input to the classifier is h_t + context_vector --> hidd_dim * 2
        self.classification_layer = nn.Linear(hidd_dim * 2, output_dim)


    def forward(self, trg_seqs, encoder_outputs, decoder_h_t, context_vectors, attention_mask):
        # trg_seqs shape: [batch_size]
        batch_size = trg_seqs.shape[0]

        # Step 1: embedding the target seqs
        embedded_seqs = self.embedding_layer(trg_seqs)
        # embedded_seqs shape: [batch_size, embed_dim]

        # concatenating the embedded trg sequence with the context_vectors
        rnn_input = torch.cat((embedded_seqs, context_vectors), dim=1)
        # rnn_input shape: [batch_size, embed_dim + hidd_dim]

        # Step 2: feeding the input to the rnn and updating the decoder_h_t
        decoder_h_t = self.rnn(rnn_input, decoder_h_t)
        # decoder_h_t shape: [batch_size, hidd_dim]

        # Step 3: updating the context vectors through attention
        context_vectors, atten_scores = self.attention(key_vectors=encoder_outputs, 
                                                       query_vector=decoder_h_t,
                                                       mask=attention_mask)

        # concatenating decoder_h_t with the context_vectors to create a 
        # prediction vector
        predictions_vector = torch.cat((decoder_h_t, context_vectors), dim=1)
        # predictions_vector: [batch_size, hidd_dim * 2]

        # Step 4: feeding the prediction vector to the fc layer
        # to a make a prediction
        prediction = self.classification_layer(predictions_vector)
        # prediction shape: [batch_size, output_dim]

        return prediction, decoder_h_t, atten_scores, context_vectors

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
    """Seq2Seq model"""
    def __init__(self, encoder_input_dim, encoder_embed_dim,
                 encoder_hidd_dim, decoder_input_dim,
                 decoder_embed_dim, decoder_output_dim,
                 src_padding_idx=0, trg_padding_idx=0, trg_sos_idx=2):

        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim=encoder_input_dim,
                               embed_dim=encoder_embed_dim,
                               hidd_dim=encoder_hidd_dim,
                               padding_idx=src_padding_idx)

        self.decoder_hidd_dim = encoder_hidd_dim * 2

        self.attention = AdditiveAttention(encoder_hidd_dim=encoder_hidd_dim,
                                           decoder_hidd_dim=self.decoder_hidd_dim)

        self.decoder = Decoder(input_dim=decoder_input_dim,
                               embed_dim=decoder_embed_dim,
                               hidd_dim=self.decoder_hidd_dim,
                               output_dim=decoder_input_dim,
                               attention=self.attention,
                               padding_idx=trg_padding_idx)

        self.src_padding_idx = src_padding_idx
        self.trg_sos_idx = trg_sos_idx
        self.sampling_temperature = 3

    def create_mask(self, src_seqs, src_padding_idx):
        mask = (src_seqs != src_padding_idx)
        return mask

    def forward(self, src_seqs, src_seqs_lengths, trg_seqs, teacher_forcing_prob=0.3):
        # trg_seqs shape: [batch_size, trg_seqs_length]
        # reshaping to: [trg_seqs_length, batch_size]
        trg_seqs = trg_seqs.permute(1, 0)
        trg_seqs_length, batch_size = trg_seqs.shape

        # passing the src to the encoder
        encoder_outputs, encoder_hidd = self.encoder(src_seqs, src_seqs_lengths)

        # creating attention masks
        attention_mask = self.create_mask(src_seqs, self.src_padding_idx)

        predictions = []
        decoder_attention_scores = []

        # initializing the trg_seqs to <s> token
        y_t = torch.ones(batch_size, dtype=torch.long) * self.trg_sos_idx

        # intializing the context_vectors to zero
        context_vectors = torch.zeros(batch_size, self.decoder_hidd_dim)

        # initializing the hidden state of the decoder to the encoder hidden state
        decoder_h_t = encoder_hidd

        # moving y_t and context_vectors to the right device
        y_t = y_t.to(encoder_hidd.device)
        context_vectors = context_vectors.to(encoder_hidd.device)

        for i in range(trg_seqs_length):
            teacher_forcing = np.random.random() < teacher_forcing_prob
            # if teacher_forcing, use ground truth target tokens
            # as an input to the decoder
            if teacher_forcing:
                y_t = trg_seqs[i]

            # do a single decoder step
            prediction, decoder_h_t, atten_scores, context_vectors = self.decoder(y_t,
                                                                                  encoder_outputs,
                                                                                  decoder_h_t,
                                                                                  context_vectors,
                                                                                  attention_mask=attention_mask)

            # If not teacher force, use the maximum 
            # prediction as an input to the decoder in 
            # the next time step
            if not teacher_forcing:
                # we multiply the predictions with a sampling_temperature
                # to make the probablities peakier, so we can be confident about the
                # maximum prediction
                pred_output_probs = F.softmax(prediction * self.sampling_temperature, dim=1)
                y_t = torch.argmax(pred_output_probs, dim=1)

            predictions.append(prediction)
            decoder_attention_scores.append(atten_scores)

        predictions = torch.stack(predictions)
        # predictions shape: [trg_seq_len, batch_size, output_dim]
        predictions = predictions.permute(1, 0, 2)
        # predictions shape: [batch_size, trg_seq_len, output_dim]

        decoder_attention_scores = torch.stack(decoder_attention_scores)
        # attention_scores_total shape: [trg_seq_len, batch_size, src_seq_len]
        decoder_attention_scores = decoder_attention_scores.permute(1, 0, 2)
        # attention_scores_total shape: [batch_size, trg_seq_len, src_seq_len]

        return predictions, decoder_attention_scores
