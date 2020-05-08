#!/bin/bash
#SBATCH -p condo 
# Set number of nodes to run
#SBATCH --nodes=1
# Set number of tasks to run
#SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=2 
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

export EXPERIMENT_NAME=M.to.F
export SYSTEM_HYP=/home/ba63/gender-bias/models/new_decoder_inferences/$EXPERIMENT_NAME/dev_preds_improv_256_128_gender_1e-6_wout_trg.inf
export GOLD_DATA=/home/ba63/gender-bias/data/christine_2019/Arabic-parallel-gender-corpus/edits_annotations/D-set-dev.$EXPERIMENT_NAME.edits_annotation

python /home/ba63/m2scorer/scripts/m2scorer.py $SYSTEM_HYP $GOLD_DATA > new_decoder_inferences/$EXPERIMENT_NAME/dev_preds_improv_256_128_gender_1e-6_wout_trg.inf.eval
