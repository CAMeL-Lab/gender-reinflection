#!/bin/bash
#SBATCH -p nvidia 
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=100GB
# Walltime format hh:mm:ss
#SBATCH --time=23:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

export DATA_DIR=data/alhafni

python main.py \
 --data_dir $DATA_DIR \
 --vectorizer_path saved_models/vectorizer.json \
 --analyzer_db_path /scratch/ba63/databases/calima-msa/calima-msa.0.2.2.utf8.db \
 --embed_trg_gender \
 --trg_gender_embedding_dim 10 \
 --use_morph_features \
 --morph_features_path saved_models/morph_features_top_1_analyses.json \
 --cache_files \
 --num_train_epochs 50 \
 --embedding_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --batch_size 32 \
 --use_cuda \
 --seed 21 \
 --do_train \
 --dropout 0.2 \
 --clip_grad 1.0 \
 --visualize_loss \
 --model_path saved_models/joint_models/joint+morph.pt
