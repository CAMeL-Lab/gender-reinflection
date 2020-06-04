import argparse
from data_utils import Vocabulary
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_examples(data_dir):
    logger.info(f'Reading Examples from {data_dir}')
    with open(data_dir, mode='r', encoding='utf8') as f:
        return f.readlines()

def write_examples(examples, data_dir):
    logger.info(f'Writing {len(examples)} examples to {data_dir}')
    with open(data_dir, mode='w', encoding='utf8') as f:
        for example in examples:
            f.write(example)
            f.write('\n')

def build_vocab(src_examples, trg_examples):
    logger.info(f'Building the Vocab...')
    src_vocab_verbs = Vocabulary()
    trg_vocab_verbs = Vocabulary()
    src_trg_adj_pairs = set()
    # we should have the same number of example on src and trg
    assert len(src_examples) == len(trg_examples)

    for i in range(len(src_examples)):
        src_example = src_examples[i]
        trg_example = trg_examples[i]

        src_example = src_example.strip().split(' ')
        if len(src_example) == 2:
            src_verb, src_adj = src_example[0], src_example[1]
        elif len(src_example) == 3:
            src_verb = src_example[0] + ' ' + src_example[1]
            src_adj = src_example[2]

        #src_vocab_adjs.append(src_adj)

        trg_example = trg_example.strip().split(' ')

        if len(trg_example) == 2:
            trg_verb, trg_adj = trg_example[0], trg_example[1]
        elif len(trg_example) == 3:
            trg_verb = trg_example[0] + ' ' + trg_example[1]
            trg_adj = trg_example[2]

        src_trg_adj_pairs.add((src_adj, trg_adj))
        #trg_vocab_adjs.append(trg_adj)
        src_vocab_verbs.add_token(src_verb)
        trg_vocab_verbs.add_token(trg_verb)

    # note: trg_vocab_verbs should be the same as src_vocab_verbs
    assert list(trg_vocab_verbs.idx_to_token.values()) == list(src_vocab_verbs.idx_to_token.values())
    #assert len(trg_vocab_adjs) == len(src_vocab_adjs)
    logger.info('Done building vocab')

    print('src verbs vocab', flush=True)
    print(list(src_vocab_verbs.idx_to_token.values()), flush=True)
    print('trg verbs vocab', flush=True)
    print(list(trg_vocab_verbs.idx_to_token.values()), flush=True)
    print(f'src adjs pair vocab size {len(src_trg_adj_pairs)}')
    print(list(src_trg_adj_pairs), flush=True)
    #print(f'trg adjs vocab size {len(trg_vocab_adjs)}')
    #return src_vocab_verbs, src_vocab_adjs, trg_vocab_verbs, trg_vocab_adjs
    return src_vocab_verbs, trg_vocab_verbs, src_trg_adj_pairs

def select_examples(src_vocab_verbs, trg_vocab_verbs, src_trg_adj_pairs):

    src_examples = []
    trg_examples = []
    src_trg_adj_pairs = list(src_trg_adj_pairs)
    src_verb_tokens = list(src_vocab_verbs.idx_to_token.values())
    trg_verb_tokens = list(trg_vocab_verbs.idx_to_token.values())
    assert src_verb_tokens == trg_verb_tokens
    # building src and trg adjs pairs
    # randomly shuffle the pairs
    random.shuffle(src_trg_adj_pairs)
    for src_adj, trg_adj in src_trg_adj_pairs:
        # randomly pick a verb
        verb = random.choice(src_verb_tokens)
        src_example = verb + ' ' + src_adj
        trg_example = verb + ' ' + trg_adj
        src_examples.append(src_example)
        trg_examples.append(trg_example)

    return src_examples, trg_examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_input_file',
        default=None,
        type=str,
        help='Path to the input src file'
        )
    parser.add_argument(
        '--trg_input_file',
        default=None,
        type=str,
        help='Path to the input trg file'
        )
    parser.add_argument(
        '--src_output_file',
        default=None,
        type=str,
        help='Path to the src output file'
        )
    parser.add_argument(
        '--trg_output_file',
        default=None,
        type=str,
        help='Path to the trg output file'
        )
    parser.add_argument(
        '--seed',
        default=21,
        type=int,
        help='Random seed'
        )
    args = parser.parse_args()
    random.seed(args.seed)

    src_examples = read_examples(args.src_input_file)
    trg_examples = read_examples(args.trg_input_file)

    src_vocab_verbs, trg_vocab_verbs, src_trg_adj_pairs = build_vocab(src_examples, trg_examples)
    selected_src_examples, selected_trg_examples = select_examples(src_vocab_verbs, trg_vocab_verbs, src_trg_adj_pairs)

    write_examples(selected_src_examples, args.src_output_file)
    write_examples(selected_trg_examples, args.trg_output_file)

if __name__ == '__main__':
    main()
