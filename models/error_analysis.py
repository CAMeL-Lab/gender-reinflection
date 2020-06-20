from data_utils import RawDataset
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_lines(data_dir):
    with open(data_dir, mode='r', encoding='utf8') as f:
        return f.readlines()

def format_tags(s):
    """Formatting for pretty printing"""
    return ["{:<" + str(len(s[i])) + "}" for i in range(len(s) - 1, -1, -1)]

def get_state_tags(src, trg):
    """Compares the src to the trg
    Args:
        - src (str): the src sentence
        - trg (str): the trg sentence

    Returns:
        - tags (list): list of tags (S or C)
    """
    tags = []
    for i, src_w in enumerate(src):
        if i > len(trg) - 1:
            break
        trg_w = trg[i]
        if src_w == trg_w:
            tags.append('S')
        else:
            tags.append('C')

    # if we over or under generated
    if len(tags) != len(src):
        print('Under-generation case:')
        print(f'src:\t\t\t{" ".join(src)}', flush=True)
        print(f'pred:\t\t\t{" ".join(trg)}', flush=True)
        print('\n',flush=True)

    if len(tags) != len(trg):
        print('Over-generation case:')
        print(f'src:\t\t\t{" ".join(src)}', flush=True)
        print(f'pred:\t\t\t{" ".join(trg)}', flush=True)
        print('\n',flush=True)

    # penalize over or under generation
    while len(tags) != len(src):
        tags.append('C')

    assert len(tags) == len(src)
    return tags


def get_eval_tags(trg, pred):
    """Compares the pred to the trg
    Args:
        - trg (list): the target sequence (either a list of words
        or a list of tags S or C)
        - pred (list): the predicted sequence (either a list of words
        or a list of tags S or C)

    Returns:
        - tags (list): list of tags (G or B)
    """

    tags = []
    for i, trg_w in enumerate(trg):
        if i > len(pred) - 1:
            break
        pred_w = pred[i]
        if trg_w == pred_w:
            tags.append('G')
        else:
            tags.append('B')

    # penalize over or under generation
    while len(tags) != len(trg):
        tags.append('B')

    return tags


def error_analysis(predictions, gold_data):
    """Does the error analysis
    Args:
        - predictions (list): list of predictions
        - gold_data (list): list of InputExample objects
    """
    reinflect_stats = {}
    eval_stats = {}
    for i, example in enumerate(gold_data):
        src = example.src
        trg = example.trg
        src_label = example.src_label
        trg_label = example.trg_label
        trg_gender = example.trg_gender

        prediction = predictions[i].strip()

        res = "CORRECT!" if trg == prediction else "INCORRECT!"
        # computing the state tags between the src and the trg
        t_tags = get_state_tags(src=src.split(' '),
                                trg=trg.split(' '))

        # computing the state tags between the src and the prediction
        p_tags = get_state_tags(src=src.split(' '),
                                trg=prediction.split(' '))

        # computing the eval tags between trg and prediction 
        eval_tags = get_eval_tags(trg=trg.split(' '),
                                  pred=prediction.split(' '))

        # computing the eval tags between the t_tags and the p_tags
        change_tags = get_eval_tags(trg=t_tags,
                                    pred=p_tags)

        # formatting
        s_format = format_tags(src.split(' '))

        # computing the stats
        for i, t_tag in enumerate(t_tags):
            p_tag = p_tags[i]
            c_tag = change_tags[i]
            e_tag = eval_tags[i]
            reinflect_stats[(t_tag, p_tag, c_tag)] = 1 + reinflect_stats.get((t_tag, p_tag, c_tag), 0)
            eval_stats[(t_tag, p_tag, e_tag)] = 1 + eval_stats.get((t_tag, p_tag, e_tag), 0)

        logger.info(f'src:\t\t\t{src}')
        logger.info(f'trg:\t\t\t{trg}')
        logger.info('t_tags:\t\t\t'+" ".join(s_format).format(*t_tags[::-1]))
        logger.info(f'pred:\t\t\t{prediction}')
        logger.info('p_tags:\t\t\t'+" ".join(s_format).format(*p_tags[::-1]))
        logger.info('c_tags:\t\t\t'+" ".join(s_format).format(*change_tags[::-1]))
        logger.info('e_tags:\t\t\t'+" ".join(s_format).format(*eval_tags[::-1]))
        logger.info(f'src label:\t\t{src_label}')
        logger.info(f'trg label:\t\t{trg_label}')
        logger.info(f'trg gender:\t\t{trg_gender}')
        logger.info(f'res:\t\t\t{res}')
        logger.info('\n\n')

    logger.info('\n\n\n')

    assert sum(list(reinflect_stats.values())) == sum(list(eval_stats.values()))

    logger.info('\t\tReinflection\t\t\tEvaluation')
    logger.info('\t\tGood\tBad\t\t\tGood\tBad')
    for k in reinflect_stats:
        t_tag = k[0]
        p_tag = k[1]
        logger.info(k[0] + '-' + k[1] + '\t' 
                    + str(reinflect_stats.get((t_tag, p_tag, 'G'), 0)) 
                    + '\t' + str(reinflect_stats.get((t_tag, p_tag, 'B'), 0))
                    + '\t\t\t' + str(eval_stats.get((t_tag, p_tag, 'G'), 0))
                    + '\t' + str(eval_stats.get((t_tag, p_tag, 'B'), 0)))

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
        error_analysis(predictions=inference_data, gold_data=gold_data.dev_examples)
    elif args.inference == 'test':
        error_analysis(predictions=inference_data, gold_data=gold_data.test_examples)

if __name__ == "__main__":
    main()
