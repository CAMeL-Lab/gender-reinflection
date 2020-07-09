from utils.data_utils import RawDataset
from collections import defaultdict
import argparse
import logging
import operator
from utils.metrics import accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_ngrams(sentence, pad_right=False, pad_left=False, ngrams=1):
    """
    Args:
     - sentence (str): a list of words
     - ngrams (int): 2 for bigrams, 3 for trigrams, etc..
     - pad_right (bool): adding </s> to the end of sentence
     - pad_left (bool): adding <s> to the beginning of sentence

    Returns:
     - ngrams of the sentence (list of tuples)
    """

    if pad_right:
        sentence = sentence + ['</s>'] * (ngrams - 1)
    if pad_left:
        sentence = ['<s>'] * (ngrams - 1) + sentence
    return [tuple(sentence[i - (ngrams - 1): i + 1]) for i in range(ngrams - 1, len(sentence))]

class MLE:
    """MLE to model P(t_w | s_w, t_g)"""

    def __init__(self, model, ngrams):
        self.model = model
        self.ngrams = ngrams

    @classmethod
    def build_model(cls, examples, ngrams=1):
        """
        Args:
            - examples (list): list of InputExample objects
            - ngrams (int): number of ngrams
        Returns:
            - mle model (default dict): The mle model where the
            keys are (sw, trg_gender) and vals are trg_w
        """

        model = defaultdict(lambda: defaultdict(lambda: 0))
        context_counts = dict()
        for i, ex in enumerate(examples):

            src = ex.src
            trg = ex.trg
            trg_g = ex.trg_gender
            src = src.split(' ')
            trg = trg.split(' ')

            # getting counts of all ngrams
            # until ngrams == 1
            for i in range(ngrams):
                src_ngrams = build_ngrams(src, ngrams=i + 1, pad_left=True)
                for j, trg_w in enumerate(trg):
                    # counts of (t_w, s_w, t_g)
                    model[(src_ngrams[j], trg_g)][trg_w] += 1
                    # counts of (s_w, t_g)
                    context_counts[(src_ngrams[j], trg_g)] = 1 + context_counts.get((src_ngrams[j], trg_g), 0)

        # turning the counts into probs
        for sw, trg_g in model:
            for trg_w in model[(sw, trg_g)]:
                model[(sw, trg_g)][trg_w] = float(model[(sw, trg_g)][trg_w]) / context_counts[(sw, trg_g)]

        return cls(model, ngrams)

    def __getitem__(self, sw_tg):
        context, trg_gender = sw_tg[0], sw_tg[1]
        # keep backing-off until a context is found
        for i in range(self.ngrams):
            if (context[i:], trg_gender) in self.model:
                return dict(self.model[(context[i:], trg_gender)])

        # worst case, pass the word as it is
        return {context[-1]: 0.0}


    def __len__(self):
        return len(self.model)

def reinflect(model, src_sentence, trg_g, ngrams=1):
    """
    Reinflects a sentence based on the mle model.
    At each time step, the model will pick the word with maximum prob.
    Args:
    - src_sentence (str): the source sentence
    - trg_g (str): the target gender
    """
    src = src_sentence.split(' ')
    src_ngrams = build_ngrams(src, ngrams=ngrams, pad_left=True)
    target = []
    for sw in src_ngrams:
        candidates = model[(sw, trg_g)]
        # print(candidates)
        argmax = max(candidates.items(), key=operator.itemgetter(1))[0]
        target.append(argmax)

    return ' '.join(target)


def inference(model, data_examples, args):
    """Does inference on a set of examples
    given a model.

    Args:
        - model (MLE): mle model
        - data_examples (list): list of InputExample objects
    """
    output_file = open(args.preds_dir + '.inf', mode='w', encoding='utf8')
    stats = {}
    mle_acc = 0

    for example in data_examples:
        src = example.src
        trg = example.trg
        trg_gender = example.trg_gender
        src_label = example.src_label
        trg_label = example.trg_label
        inference = reinflect(model=model, src_sentence=src, trg_g=trg_gender, ngrams=args.ngrams)
        mle_acc += accuracy(trg=trg, pred=inference)
        correct = 'CORRECT!' if trg == inference else 'INCORRECT!'
        if inference == trg:
            stats[(src_label, trg_label, 'correct')] = 1 + stats.get((src_label, trg_label, 'correct'), 0)
        else:
            stats[(src_label, trg_label, 'incorrect')] = 1 + stats.get((src_label, trg_label, 'incorrect'), 0)

        logger.info(f'src:\t\t\t{src}')
        logger.info(f'trg:\t\t\t{trg}')
        logger.info(f'pred:\t\t\t{inference}')
        logger.info(f'src label:\t\t{src_label}')
        logger.info(f'trg label:\t\t{trg_label}')
        logger.info(f'trg gender:\t\t{trg_gender}')
        logger.info(f'res:\t\t\t{correct}')
        logger.info('\n\n')
        output_file.write(inference)
        output_file.write('\n')

    mle_acc /= len(data_examples)

    output_file.close()
    logger.info('*******STATS*******')
    total_examples = sum([stats[x] for x in stats])
    logger.info(f'TOTAL EXAMPLES: {total_examples}')
    logger.info('\n')
    correct_inferneces = {(x[0], x[1]): stats[x] for x in stats if x[2] == 'correct'}
    incorrect_inferneces = {(x[0], x[1]): stats[x] for x in stats if x[2] == 'incorrect'}

    total_correct = sum([v for k,v in correct_inferneces.items()])
    total_incorrect = sum([v for k, v in incorrect_inferneces.items()])

    logger.info('Results:')
    for x in correct_inferneces:
        logger.info(f'{x[0]}->{x[1]}')
        logger.info(f'\tCorrect: {correct_inferneces.get(x, 0)}\tIncorrect: {incorrect_inferneces.get(x, 0)}')

    logger.info(f'--------------------------------')
    logger.info(f'Total Correct: {total_correct}\tTotal Incorrect: {total_incorrect}')
    logger.info(f'Accuracy:\t{mle_acc}')

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
        "--ngrams",
        type=int,
        default=1,
        help="The MLE model ngrams."
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
        required=True,
        help="The directory to write the translations to"
    )

    args = parser.parse_args()

    # reading the data
    raw_data = RawDataset(args.data_dir)

    # building the MLE model based on the training examples
    mle_model = MLE.build_model(raw_data.train_examples, ngrams=args.ngrams)

    if args.inference_mode == 'dev':
        inference(mle_model, raw_data.dev_examples, args)
    elif args.inference_mode == 'test':
        inference(mle_model, raw_data.test_examples, args)

if __name__ == "__main__":
    main()
