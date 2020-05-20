#!/bin/bash
#SBATCH -p nvidia 
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=30GB
# Walltime format hh:mm:ss
#SBATCH --time=23:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

export EXPERIMENT_NAME=M.to.F
export DATA_DIR=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/

python main_fin.py \
 --data_dir $DATA_DIR \
 --vectorizer_path /home/ba63/gender-bias/models/saved_models/vectorizer.json \
 --analyzer_db_path /scratch/ba63/databases/calima-msa/calima-msa.0.2.2.utf8.db \
 --morph_features_path /home/ba63/gender-bias/models/saved_models/morph_features.json \
 --use_morph_features \
 --embed_trg_gender \
 --trg_gender_embedding_dim 50 \
 --use_fasttext_embeddings \
 --fasttext_embeddings_kv_path /scratch/ba63/gender_bias/data/OpenSubtitles_fasttext_embeddings.kv \
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
 --visualize_loss \
 --model_path /home/ba63/gender-bias/models/saved_models/$EXPERIMENT_NAME/char_level_improv_256_128_gender_1e-6_w_trg_fin_2_new_fasttext_space_1e-3.pt
