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

export SRC_INPUT_FILE=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/S-set.M
export TRG_INPUT_FILE=/scratch/ba63/gender_bias/data/christine_2019/Arabic-parallel-gender-corpus/S-set.F
export SRC_OUTPUT_FILE=/home/ba63/test.M
export TRG_OUTPUT_FILE=/home/ba63/test.F

python preprocess_syn_data.py --src_input_file $SRC_INPUT_FILE --trg_input_file $TRG_INPUT_FILE --src_output_file $SRC_OUTPUT_FILE --trg_output_file $TRG_OUTPUT_FILE
