from nmt_sampler import NMT_Batch_Sampler
from queue import PriorityQueue
import operator
import torch
import torch.nn.functional as F
import re

class BeamSearchNode:
    """A class to represent the node during the beam search"""
    def __init__(self, hidd_state, prev_node, word_idx, log_prob, length):
        """
        :param hidd_state: decoder hidden state
        :param prev_node: the previous node (parent)
        :param word_idx: the word index
        :param log_prob: the log probability
        :param length: length of decoded sentence
        """
        self.h = hidd_state
        self.prevNode = prev_node
        self.wordid = word_idx
        self.logp = log_prob
        self.leng = length

    def eval(self, alpha=1):
        reward = 0
        # Add here a function for shaping a reward
        # the log prob will be normalized by the length of the sentence
        # as defined by Wu et. al: https://arxiv.org/pdf/1609.08144.pdf
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
        #return self.logp / float(self.leng)**alpha

    def __lt__(self, other):
       """Overriding the less than function to handle
       the case if two nodes have the same log_prob so
       they can fit in the priority queue"""
       return self.logp < other.logp

class BeamSampler(NMT_Batch_Sampler):
    """A subclass of NMT_Batch_Sampler that uses beam_search for decoding"""
    def __init__(self, model, src_vocab_char,
                 src_vocab_word, trg_vocab_char,
                 src_labels_vocab, trg_labels_vocab,
                 trg_gender_vocab):

        super(BeamSampler, self).__init__(model, src_vocab_char,
                                          src_vocab_word, trg_vocab_char,
                                          src_labels_vocab, trg_labels_vocab,
                                          trg_gender_vocab
                                          )

    def beam_decode(self, sentence, trg_gender=None, topk=3, beam_width=5, max_len=512):
        """
        :param sentence: the source sentence
        :param topk: number of sentences to generate from beam search. Defaults to 3
        :param beam_width: the beam size. If 1, then we do greed search. Defaults to 5
        :param max_len: the maximum length of the decoded sentence. Defaults to 512
        :returns: decoded_sentences: list of tuples. Each tuple is (log_prob, decoded_sentence)
        """

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
        decoder_hidden = encoder_h_t
        #decoder_hidden = torch.tanh(self.model.linear_map(encoder_h_t))

        context_vectors = torch.zeros(1, self.model.encoder.rnn.hidden_size * 2)

        # if beam_width == 1, then we're doing greedy decoding
        beam_width = beam_width

        # number of candidates to generate.
        topk = topk

        # topk must be <= beam_width
        if topk > beam_width:
            raise Exception("topk candidates must be <= beam_width")

        decoded_batch = []

        # starting input to the decoder is the <s> token
        decoder_input = torch.LongTensor([self.trg_vocab_char.sos_idx])

        # number of sentences to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        # each element in the queue will be (-log_prob, beam_node)
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while max_len > 0:
            max_len -= 1
            # give up when decoding takes too long
            if qsize > 20000:
                print('hiiii')
                break

            # fetch the best node (i.e. node with minimum negative log prob)
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            # if we predict the </s> token, this means we finished decoding a sentence
            if n.wordid.item() == self.trg_vocab_char.eos_idx and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required, stop beam search
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            with torch.no_grad():
                decoder_output, decoder_hidden, atten_scores, context_vectors = self.model.decoder(trg_seqs=decoder_input,
                                                                                                   encoder_outputs=encoder_outputs,
                                                                                                   decoder_h_t=decoder_hidden,
                                                                                                   context_vectors=context_vectors,
                                                                                                   attention_mask=attention_mask,
                                                                                                   trg_gender=vectorized_trg_gender
                                                                                                   )

            # obtaining log probs from the decoder predictions
            decoder_output = F.log_softmax(decoder_output, dim=1)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            # indexes shape: [batch_size, beam_width]
            # log_prob shape: [batch_size, beam_width]

            # expanding the current beam (n)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].unsqueeze(0)
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put the expanded beams in the queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))

            # increase qsize
            qsize += len(nextnodes) - 1

        # choose topk beams
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        # sorting the topk beams by their negative log probs
        endnodes = sorted(endnodes, key=lambda x: x[0])

        # decoding
        #TODO: Decoding currently works for one sentence at a time,
        #Bashar needs to make it work on the batch
        decoded_sentences = []
        for score, n in endnodes:
            decoded_sentence = []
            decoded_sentence.append(n.wordid.item())
            # backtrack 
            while n.prevNode != None:
                n = n.prevNode
                decoded_sentence.append(n.wordid.item())
            # reversing the decoding
            decoded_sentence = decoded_sentence[::-1]
            decoded_sentences.append((score, decoded_sentence))

        str_decoded_sentence = self.get_str_sentence(decoded_sentences[0][1], self.trg_vocab_char)
        return str_decoded_sentence

