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
export DATA_DIR=data/alhafni
 
python utils/error_analysis.py \
 --data_dir $DATA_DIR \
 --normalized \
 --inference_mode dev \
 --inference_data logs/error_analysis/$EXPERIMENT_NAME/dev.joint+morph.inf.norm
