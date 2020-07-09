#!/bin/bash
#SBATCH -p serial 
# Set number of nodes to run
#SBATCH --nodes=1
# Set number of tasks to run
#SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=15
# memory
#SBATCH --mem=50GB
# Walltime format hh:mm:ss
#SBATCH --time=48:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

module load gcc

export INPUT=/scratch/nlp/data/Arabic_Gigaword_Fifth_Edition/gigaword.sep_punc.txt
export OUTPUT=/scratch/ba63/gender_bias/data/gigword_fasttext_embeddings

/home/ba63/fastText-0.9.1/fasttext skipgram -input $INPUT -minn 2 -maxn 5 -dim 300 -epoch 1 -output $OUTPUT
