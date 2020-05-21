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

export DATA_DIR=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/
export EXPERIMENT_NAME=M.to.F

python main_fin.py \
 --data_dir $DATA_DIR \
 --analyzer_db_path /scratch/ba63/databases/calima-msa/calima-msa.0.2.2.utf8.db \
 --use_morph_features \
 --embed_trg_gender \
 --trg_gender_embedding_dim 50 \
 --embedding_dim 128 \
 --hidd_dim 256 \
 --use_fasttext_embeddings \
 --fasttext_embeddings_kv_path /scratch/ba63/gender_bias/data/OpenSubtitles_fasttext_embeddings.kv \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --seed 21 \
 --model_path /home/ba63/gender-bias/models/saved_models/$EXPERIMENT_NAME/char_level_improv_256_128_gender_1e-6_w_trg_fin_2_new_zero_morph_fasttext_clip_norm.pt \
 --do_inference \
 --inference_mode dev \
 --preds_dir /home/ba63/gender-bias/models/dev_preds_improv_256_128_gender_1e-6_w_trg_fin_2_new_zero_morph_fasttext_clip_norm
