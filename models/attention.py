import torch
import torch.nn as nn
import torch.nn.functional as F

def GlorotLinear(input_dim, output_dim, bias=True):
    """Returns a Glorot initialized linear layer for optimal gradient flow"""
    linear = nn.Linear(input_dim, output_dim, bias=bias)
    nn.init.xavier_uniform_(linear.weight)
    if bias:
        nn.init.constant_(linear.bias, 0)
    return linear

class AdditiveAttention(nn.Module):
    """Attention mechanism as a MLP
    as used by Bahdanau et. al 2015"""

    def __init__(self, encoder_hidd_dim, decoder_hidd_dim):
        super(AdditiveAttention, self).__init__()
        self.atten = nn.Linear((encoder_hidd_dim * 2) + decoder_hidd_dim, decoder_hidd_dim)
        self.v = nn.Linear(decoder_hidd_dim, 1, bias=False)
        #self.atten = GlorotLinear((encoder_hidd_dim * 2) + decoder_hidd_dim, decoder_hidd_dim)
        #self.v = GlorotLinear(decoder_hidd_dim, 1, bias=False)


    def forward(self, key_vectors, query_vector, mask):
        """key_vectors: encoder hidden states.
           query_vector: decoder hidden state at time t
           mask: the mask vector of zeros and ones
        """

        #key_vectors shape: [batch_size, src_seq_length, encoder_hidd_dim * 2]
        #query_vector shape: [num_layers * num_dirs, batch_size, decoder_hidd_dim]

        batch_size, src_seq_length, encoder_hidd_dim = key_vectors.shape

        # applying attention to the hidden state at the last layer of the decoder
        query_vector = query_vector[-1, :, :]
        # query_vector shape: [batch_size, decoder_hidd_dim]

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
        atten_scores = atten_scores.masked_fill(mask==0, -float('inf'))

        # Step 2: normalizing atten_scores through a softmax to get probs
        atten_scores = F.softmax(atten_scores, dim=1)

        # Step 3: computing the new context vector
        context_vectors = torch.matmul(key_vectors.permute(0, 2, 1), atten_scores.unsqueeze(2)).squeeze(dim=2)

        # context_vectors shape: [batch_size, encoder_hidd_dim * 2]

        return context_vectors, atten_scores

class GeneralAttention(nn.Module):
    """General Attention mechanism
    as described by Luong et. al 2015"""

    def __init__(self, encoder_hidd_dim, decoder_hidd_dim):
        super(GeneralAttention, self).__init__()
        self.linear_map = nn.Linear(encoder_hidd_dim * 2, decoder_hidd_dim, bias=False)

    def forward(self, key_vectors, query_vector, mask):
        """key_vectors: encoder hidden states.
           query_vector: decoder hidden state at time t
           mask: the mask vector of zeros and ones
        """

        #key_vectors shape: [batch_size, src_seq_length, encoder_hidd_dim * 2]
        #query_vector shape: [num_layers * num_dirs, batch_size, decoder_hidd_dim]

        batch_size, src_seq_length, encoder_hidd_dim = key_vectors.shape

        # applying attention to the last hidden state of the decoder
        query_vector = query_vector[-1, :, :]
        # query_vector shape: [batch_size, decoder_hidd_dim]

        # mapping the key_vectors from encoder_hidd_dim * 2 to decoder_hidd_dim
        mapped_key_vectors = self.linear_map(key_vectors)
        # key_vectors shape: [batch_size, src_seq_length, decoder_hidd_dim]

        # performing the dot product
        atten_scores = torch.matmul(query_vector.unsqueeze(1), mapped_key_vectors.permute(0, 2, 1)).squeeze(1)
        # atten_scores shape: [batch_size, src_seq_len]

        # masking the atten_scores
        atten_scores = atten_scores.masked_fill(mask==0, -float('inf'))

        # Step 2: normalizing atten_scores through a softmax to get probs
        atten_scores = F.softmax(atten_scores, dim=1)

        # Step 3: computing the new context vector
        context_vectors = torch.matmul(key_vectors.permute(0, 2, 1), atten_scores.unsqueeze(2)).squeeze(dim=2)

        # context_vectors shape: [batch_size, encoder_hidd_dim * 2]

        return context_vectors, atten_scores


def DotProductAttention(query, keys, mask):
    """
    Args:
        - query: decoder hidden state
        - keys: encoder outputs (hidden states from the last layer)

    NOTE: This attention works only when encoder_hidd_dim * 2 == decoder_hidd_dim
    """
    #keys shape: [batch_size, src_seq_length, encoder_hidd_dim * 2]
    #query shape: [num_layers * num_dirs, batch_size, encoder_hidd_dim * 2]

    # applying attention on the last layer of the decoder
    query = query[-1, :, :]
    #query shape: [batch_size, encoder_hidd_dim * 2]

    attention_scores = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)
    # attention_scores shape: [batch_size, src_seq_length]

    # masking the attention_scores
    attention_scores = attention_scores.masked_fill(mask==0, -float('inf'))

    # normalizing the attention_scores through a softmax
    attention_scores = F.softmax(attention_scores, dim=1)

    # computing the context vector
    context_vector = torch.matmul(keys.permute(0, 2, 1), attention_scores.unsqueeze(-1)).squeeze(dim=2)
    #context_vector shape: [batch_size, encoder_hidd_dim * 2]

    return context_vector, attention_scores
