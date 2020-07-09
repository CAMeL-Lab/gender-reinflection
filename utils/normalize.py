import argparse
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize(lines):
    """
    Args:
        - lines (list): list of sentences
    Retturns:
        - norm_lines (list): list of normalized lines
    """
    norm_lines = []
    for line in lines:
        norm_line = line.strip()
        norm_line = normalize_alef_ar(norm_line)
        norm_line = normalize_alef_maksura_ar(norm_line)
        norm_line = normalize_teh_marbuta_ar(norm_line)
        norm_lines.append(norm_line)
    return norm_lines

def read_examples(file_path):
    with open(file_path, mode='r', encoding='utf8') as f:
        return f.readlines()

def write_examples(file_path, examples):
     with open(file_path, mode='w', encoding='utf8') as f:
        for example in examples:
            f.write(example)
            f.write('\n')
        f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
            type=str,
        default=None,
        help='The path to the input file containing the unormalized data'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='The path to the output file containing the normaized data'
    )

    args = parser.parse_args()

    examples = read_examples(args.input_file)
    logger.info(f'Normalizing {len(examples)} examples from {args.input_file}')
    norm_examples = normalize(examples)
    write_examples(args.output_file, norm_examples)
    logger.info(f'Normalized examples written to {args.output_file}')

if __name__ == "__main__":
    main()
