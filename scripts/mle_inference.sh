#!/bin/bash
#SBATCH -p condo 
# Set number of nodes to run
#SBATCH --nodes=1
# Set number of tasks to run
#SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=5 
# memory
#SBATCH --mem=10GB
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err
module purge

export DATA_DIR=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/
export EXPERIMENT_NAME=arin.to.F
 
python /home/ba63/gender-bias/mle.py \
 --data_dir $DATA_DIR \
 --inference_mode test \
 --ngrams 2 \
 --preds_dir /home/ba63/gender-bias/models/logs/joint_models/test_mle_baseline_2gram_test
