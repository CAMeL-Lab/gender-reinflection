#!/bin/bash
#SBATCH -p condo 
# Set number of nodes to run
#SBATCH --nodes=1
# Set number of tasks to run
#SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=10 
# memory
#SBATCH --mem=120GB
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

module purge

export FASTTEXT_BIN_PATH=/scratch/ba63/gender_bias/data/gigword_fasttext_embeddings.bin
export KV_OUTPUT_PATH=/scratch/ba63/gender_bias/data/gigword_fasttext_embeddings.kv

python postprocess_ft_embeddings.py --fasttext_embedding_dir $FASTTEXT_BIN_PATH --kv_output_dir $KV_OUTPUT_PATH
