from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from data_utils import RawDataset
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_lines(data_dir):
    with open(data_dir, mode='r', encoding='utf8') as f:
        return f.readlines()

def gender_id(predictions, gold_data):
    """Does the gender identification
    Args:
        - predictions (list): list of predictions
        - gold_data (list): list of InputExample objects
    """

    id_stats = {}
    gold_src_genders = []
    pred_src_genders = []

    for i, example in enumerate(gold_data):
        src = example.src
        trg = example.trg
        src_label = example.src_label
        trg_gender = example.trg_gender
        trg_label = example.trg_label
        prediction = predictions[i].strip()

        # getting the src gender
        if src == trg:
            src_gender = trg_gender
        else:
            src_gender = "M" if trg_gender == "F" else "F"

        # getting the predicted src gender
        if src == prediction:
            pred_src_gender = trg_gender
        else:
            pred_src_gender = "M" if trg_gender == "F" else "F"

        if src_gender == pred_src_gender:
            id_stats[(src_gender, pred_src_gender, 'correct')] = 1 + \
            id_stats.get((src_gender, pred_src_gender, 'correct'), 0)
        else:
            id_stats[(src_gender, pred_src_gender, 'incorrect')] = 1 + \
            id_stats.get((src_gender, pred_src_gender, 'incorrect'), 0)

        gold_src_genders.append(src_gender)
        pred_src_genders.append(pred_src_gender)

        res = "CORRECT!" if trg == prediction else "INCORRECT!"

        logger.info(f'src:\t\t\t{src}')
        logger.info(f'trg:\t\t\t{trg}')
        logger.info(f'pred:\t\t\t{prediction}')
        logger.info(f'trg gender:\t\t{trg_gender}')
        logger.info(f'src gender:\t\t{src_gender}')
        logger.info(f'pred src gender:\t\t{pred_src_gender}')
        logger.info(f'res:\t\t\t{res}')
        logger.info('\n\n')

    assert len(gold_src_genders) == len(pred_src_genders)

    total_correct = sum([v for k, v in id_stats.items() if k[2] == 'correct'])
    total_incorrect = sum([v for k, v in id_stats.items() if k[2] == 'incorrect'])

    correct_id = {(x[0], x[1]): id_stats[x] for x in id_stats if x[2] == 'correct'}
    incorrect_id = {(x[0], x[1]): id_stats[x] for x in id_stats if x[2] == 'incorrect'}

    logger.info('\n\n\n')
    logger.info('Gender ID Results:')
    # logger.info(id_stats)
    for x in id_stats:
        input_gender = x[0]
        predicted_gender = x[1]
        logger.info(f'{input_gender}->{predicted_gender}')
        logger.info(f'\tCorrect: {correct_id.get((input_gender, predicted_gender), 0)}'\
                    f'\tIncorrect: {incorrect_id.get((input_gender, predicted_gender), 0)}')

    logger.info(f'--------------------------------')
    logger.info(f'Total Correct: {total_correct}\tTotal Incorrect: {total_incorrect}')

    logger.info('\n\n\n')
    logger.info(f'Metrics:')

    accuracy = accuracy_score(y_true=gold_src_genders,
                              y_pred=pred_src_genders)

    f1 = f1_score(y_true=gold_src_genders,
                  y_pred=pred_src_genders,
                  average=None,
                  labels=["M", "F"])

    precision = precision_score(y_true=gold_src_genders,
                                y_pred=pred_src_genders,
                                average=None,
                                labels=["M", "F"])

    recall = recall_score(y_true=gold_src_genders,
                          y_pred=pred_src_genders,
                          average=None,
                          labels=["M", "F"])

    logger.info(f'Accuracy: {accuracy}')
    logger.info(f'F1_M: {f1[0]}')
    logger.info(f'F1_F: {f1[1]}')

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
        "--inference_data",
        default=None,
        type=str,
        required=True,
        help="The inference data path"
    )

    args = parser.parse_args()
    gold_data = RawDataset(args.data_dir)
    inference_data = read_lines(args.inference_data)

    if args.inference_mode == 'dev':
        gender_id(predictions=inference_data, gold_data=gold_data.dev_examples)
    elif args.inference_mode == 'test':
        gender_id(predictions=inference_data, gold_data=gold_data.test_examples)

if __name__ == "__main__":
    main()
