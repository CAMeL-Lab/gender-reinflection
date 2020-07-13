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

export DATA_SPLIT=dev
export EXPERIMENT_NAME=arin+arin.to.D-set-$DATA_SPLIT.ar.M+F
export GOLD_DATA=D-set-$DATA_SPLIT.ar.M+D-set-$DATA_SPLIT.ar.F.normalized
export EDITS_ANNOTATIONS=D-set-$DATA_SPLIT.$EXPERIMENT_NAME.edits_annotation.normalized

export SYSTEM_HYP=models/logs/joint_models_norm/dev_preds_256_128_2_layers_w_morph_top_1_analyses_w_trg_clip_norm_new_enc_new_no_bias_v_char_10_trg_no_last.inf.norm

export GOLD_ANNOTATION=data/alhafni/edits_annotations_normalized/$EDITS_ANNOTATIONS

export TRG_GOLD_DATA=data/alhafni/joint_model/$GOLD_DATA

# run M2 Scorer evaluation
eval "$(conda shell.bash hook)"
conda activate python2

m2_eval=$(python /home/ba63/m2scorer/scripts/m2scorer.py $SYSTEM_HYP $GOLD_ANNOTATION)

conda activate python3

# run accuracy evaluation
accuracy=$(python utils/metrics.py --trg_directory $TRG_GOLD_DATA --pred_directory $SYSTEM_HYP)

# run BLEU evaluation
bleu=$(cat $SYSTEM_HYP | sacrebleu $TRG_GOLD_DATA --force --short)

printf "%s\n%s\n%-12s%s" "$m2_eval" "$accuracy" "BLEU" ": $bleu" > eval
