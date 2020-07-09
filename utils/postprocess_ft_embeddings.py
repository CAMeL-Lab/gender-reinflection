from gensim.models.fasttext import FastText
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""A simple script to convert binary
fasttext embeddings to keyedvectors"""

parser = argparse.ArgumentParser()
parser.add_argument('--fasttext_embedding_dir',
                    type=str,
                    help="The path to the fasttext embeddings bin file.",
                    required=True
                   )

parser.add_argument('--kv_output_dir',
                    type=str,
                    help="The path of the KeyedVectors file.",
                    required=True
                   )

logger.info('Converting FastText Embeddings to KVs.....')
args = parser.parse_args()
fasttext_wv = FastText.load_fasttext_format(args.fasttext_embedding_dir)
kv = fasttext_wv.wv
kv.save(args.kv_output_dir)
logger.info(f'Saving KVs at {args.kv_output_dir}')
logger.info('Done!')
