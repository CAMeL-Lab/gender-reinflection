#!/bin/bash
#SBATCH -p condo 
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120000
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

module purge

export DATA_DIR=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/

python main.py \
 --data_dir $DATA_DIR \
 --vectorizer_path /home/ba63/gender-bias/models/saved_models/vectorizer.json \
 --cache_files \
 --num_train_epochs 50 \
 --embedding_dim 32 \
 --hidd_dim 64 \
 --learning_rate 5e-4 \
 --batch_size 32 \
 --use_cuda \
 --seed 21 \
 --do_train \
 --model_path /home/ba63/gender-bias/models/saved_models/char_level_nmt_old.pt
