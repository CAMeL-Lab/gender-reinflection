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

# A simple script that will compute the accuracy and BLEU scores 
# between the models generated output and the gold references 

module purge
eval "$(conda shell.bash hook)"
conda activate python3

export EXPERIMENT_NAME=arin.to.M
export SYSTEM_HYP=/home/ba63/gender-bias/models/new_decoder_inferences/$EXPERIMENT_NAME/dev_preds_improv_256_128_gender_1e-6_w_trg_fin_2_new_zero_morph_fasttext.inf
export TRG_GOLD_DATA=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/D-set-dev.ar.M


accuracy=$(python /home/ba63/gender-bias/models/metrics.py --trg_directory $TRG_GOLD_DATA --pred_directory $SYSTEM_HYP)

bleu=$(cat $SYSTEM_HYP | sacrebleu $TRG_GOLD_DATA --force --short)


printf "Accuracy: $accuracy\nBLEU: $bleu" > eval
