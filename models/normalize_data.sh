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

export DATA_DIR=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus
export SPLIT_NAME=D-set-dev.ar.F

python normalize.py --input_file $DATA_DIR/$SPLIT_NAME --output_file $DATA_DIR/$SPLIT_NAME.normalized
