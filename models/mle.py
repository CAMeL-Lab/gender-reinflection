from data_utils import RawDataset
from collections import defaultdict
import argparse
import logging
import operator
from metrics import accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLE:
    """MLE to model P(t_w | s_w, t_g)"""

    def __init__(self, model):
        self.model = model

    @classmethod
    def build_model(cls, examples):
        """
        Args:
            - examples (list): list of InputExample objects

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

            for i, trg_w in enumerate(trg):
                # counts of (t_w, s_w, t_g)
                model[(src[i], trg_g)][trg_w] += 1
                # counts of (s_w, t_g)
                context_counts[(src[i], trg_g)] = 1 + context_counts.get((src[i], trg_g), 0)

        # turning the counts into probs
        for sw, trg_g in model:
            for trg_w in model[(sw, trg_g)]:
                model[(sw, trg_g)][trg_w] = float(model[(sw, trg_g)][trg_w]) / context_counts[(sw, trg_g)]

        return cls(model)

    def __getitem__(self, sw_tg):
        if sw_tg in self.model:
            return dict(self.model[sw_tg])
        else:
            return {'<unk>': 0.0}

    def __len__(self):
        return len(self.model)

def reinflect(model, src_sentence, trg_g):
    """Reinflects a sentence based on the mle model.
    At each time step, the model will pick the word with maximum prob.

    Args:
        - src_sentence (str): the source sentence
        - trg_g (str): the target gender

    Returns:
        - reinflected sentence (str)
    """
    src = src_sentence.split(' ')
    target = []
    for sw in src:
        candidates = model[(sw, trg_g)]
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
        inference = reinflect(model=model, src_sentence=src, trg_g=trg_gender)
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
    mle_model = MLE.build_model(raw_data.train_examples)

    if args.inference_mode == 'dev':
        inference(mle_model, raw_data.dev_examples, args)
    elif args.inference == 'test':
        inference(mle_model, raw_data.test_examples, args)

if __name__ == "__main__":
    main()
