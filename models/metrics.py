import argparse

def accuracy(trg, pred):
     """
     Args:
        - trg (str): reference
        - pred (str): generated output
     Returns:
        - acc (float): word level accuracy
     """
     trg_words = trg.split(' ')
     pred_words = pred.split(' ')
     acc = 0
     for i, w in enumerate(trg_words):
         if i < len(pred_words):
             if w == pred_words[i]:
                 acc += 1
         else:
             break
     return float(acc) / float(len(trg_words))

def corpus_accuracy(trg_corpus, pred_corpus):
    """
    Args:
        - trg_corpus (list): list of references
        - pred_corpus (list): list of model's predictions
    Returns:
        - corpus_acc (float): average accuracy accross the corpus
    """
    corpus_acc = 0
    for i, line in enumerate(trg_corpus):
        corpus_acc += accuracy(trg=trg_corpus[i], pred=pred_corpus[i])
    return corpus_acc / len(trg_corpus)

def read_examples(data_dir):
    with open(data_dir, mode='r', encoding='utf8') as f:
        return f.readlines()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trg_directory',
        default=None,
        type=str,
        help="Directory of the target corpus"
    )
    parser.add_argument(
        '--pred_directory',
        default=None,
        type=str,
        help="Directory of the prediction corpus"
    )

    args = parser.parse_args()

    trg_examples = read_examples(args.trg_directory)
    pred_examples = read_examples(args.pred_directory)
    assert len(trg_examples) == len(pred_examples)

    accuracy = corpus_accuracy(trg_corpus=trg_examples,
                               pred_corpus=pred_examples
                              )
    print(accuracy)

if __name__ == "__main__":
    main()
