#!/bin/bash
#SBATCH -p condo 
# Set number of nodes to run
#SBATCH --nodes=1
# Set number of tasks to run
#SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=10 
# memory
#SBATCH --mem=10GB
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err
module purge

export EXPERIMENT_NAME=arin.to.F
export DATA_DIR=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/
 
python /home/ba63/gender-bias/utils/error_analysis.py \
 --data_dir $DATA_DIR \
 --inference_mode dev \
 --inference_data /home/ba63/gender-bias/models/logs/joint_models_norm/$EXPERIMENT_NAME/dev_preds_256_128_2_layers_w_morph_top_1_analyses_w_trg_clip_norm_new_enc_new_no_bias_v_char_10_trg_no_last.inf.norm


#dev_preds_256_128_2_layers_w_trg_clip_norm_new_enc_new_no_bias_v_char_10_trg_no_last.inf.norm
