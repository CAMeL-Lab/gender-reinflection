#!/bin/bash
#SBATCH -p nvidia 
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=20GB
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

export DATA_DIR=data/alhafni

python main.py \
 --data_dir $DATA_DIR \
 --embed_trg_gender \
 --analyzer_db_path /scratch/ba63/databases/calima-msa/calima-msa.0.2.2.utf8.db \
 --use_morph_features \
 --trg_gender_embedding_dim 10 \
 --embedding_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --seed 21 \
 --model_path saved_models/joint_models/joint+morph.pt \
 --do_inference \
 --inference_mode dev \
 --preds_dir logs/reinflection/joint_models/dev.joint+morph_blabla
