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
eval "$(conda shell.bash hook)"
conda activate python2

export EXPERIMENT_NAME=arin+arin.to.D-set-test.ar.M+F

export DEV_SET=D-set-test.ar.M+D-set-test.ar.F

export SYSTEM_HYP=/home/ba63/gender-bias/models/logs/joint_models_norm/test_preds_256_128_2_layers_w_morph_top_1_analyses_w_trg_clip_norm_new_enc_new_no_bias_v_char_10_trg_no_last_test.inf.norm
#export SYSTEM_HYP=/home/ba63/gender-bias/models/logs/joint_models_norm/test_preds_256_128_2_layers_w_morph_top_1_analyses_w_trg_clip_norm_new_enc_new_no_bias_v_char_10_trg_no_last.inf.norm

#export SYSTEM_HYP=/home/ba63/gender-bias/models/logs/joint_models_norm/$EXPERIMENT_NAME/mle_baseline_5gram.inf.norm

export GOLD_ANNOTATION=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/edits_annotations_normalized/D-set-test.$EXPERIMENT_NAME.edits_annotation.normalized

export TRG_GOLD_DATA=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/joint_model/$DEV_SET.normalized

# run M2 Scorer evaluation
m2_eval=$(python /home/ba63/m2scorer/scripts/m2scorer.py $SYSTEM_HYP $GOLD_ANNOTATION)

conda activate python3

# run accuracy evaluation
accuracy=$(python /home/ba63/gender-bias/models/metrics.py --trg_directory $TRG_GOLD_DATA --pred_directory $SYSTEM_HYP)

# run BLEU evaluation
bleu=$(cat $SYSTEM_HYP | sacrebleu $TRG_GOLD_DATA --force --short)

printf "%s\n%s\n%-12s%s" "$m2_eval" "$accuracy" "BLEU" ": $bleu" > eval
