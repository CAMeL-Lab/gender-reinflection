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

export EXPERIMENT_NAME=arin.to.M

export DEV_SET=D-set-dev.ar.M

export SYSTEM_HYP=/home/ba63/gender-bias/models/new_decoder_inferences/$EXPERIMENT_NAME/dev_preds_improv_256_128_gender_1e-6_w_trg_fin_2_new.inf

export GOLD_ANNOTATION=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/edits_annotations/D-set-dev.$EXPERIMENT_NAME.edits_annotation

export TRG_GOLD_DATA=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/$DEV_SET

# run M2 Scorer evaluation
m2_eval=$(python /home/ba63/m2scorer/scripts/m2scorer.py $SYSTEM_HYP $GOLD_ANNOTATION)

conda activate python3

# run accuracy evaluation
accuracy=$(python /home/ba63/gender-bias/models/metrics.py --trg_directory $TRG_GOLD_DATA --pred_directory $SYSTEM_HYP)

# run BLEU evaluation
bleu=$(cat $SYSTEM_HYP | sacrebleu $TRG_GOLD_DATA --force --short)

printf "%s\n%-12s%s\n%-12s%s" "$m2_eval" "Accuracy" ": $accuracy" "BLEU" ": $bleu" > eval
