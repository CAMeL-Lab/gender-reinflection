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

export INPUT=/scratch/ba63/gender_bias/data/OpenSubtitles.ar-en.ar.sep.punc
export OUTPUT=/scratch/ba63/gender_bias/data/fasttext_embeddings/OpenSubtitles_fasttext_embeddings_100

/home/ba63/fastText-0.9.1/fasttext skipgram -input $INPUT -minn 2 -maxn 5 -dim 100 -epoch 10 -output $OUTPUT
