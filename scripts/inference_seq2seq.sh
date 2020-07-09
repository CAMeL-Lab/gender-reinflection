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
export EXPERIMENT_NAME=arin.to.M

#--use_morph_features \
#--use_fasttext_embeddings \
#--fasttext_embeddings_kv_path /scratch/ba63/gender_bias/data/fasttext_embeddings/OpenSubtitles_fasttext_embeddings_100.kv \

python /home/ba63/gender-bias/main.py \
 --data_dir $DATA_DIR \
 --analyzer_db_path /scratch/ba63/databases/calima-msa/calima-msa.0.2.2.utf8.db \
 --use_morph_features \
 --embed_trg_gender \
 --trg_gender_embedding_dim 10 \
 --embedding_dim 128 \
 --hidd_dim 256 \
 --num_layers 2 \
 --learning_rate 5e-4 \
 --seed 21 \
 --model_path /home/ba63/gender-bias/models/saved_models/joint_models/char_level_256_128_2_layers_w_morph_top_1_analyses_w_trg_clip_norm_new_enc_new_no_bias_v_char_10_trg_no_last.pt \
 --do_inference \
 --inference_mode dev \
 --preds_dir /home/ba63/gender-bias/models/logs/joint_models/dev_preds_256_128_2_layers_w_morph_top_1_analyses_w_trg_clip_norm_new_enc_new_no_bias_v_char_10_trg_no_last_test
