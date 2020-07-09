#!/bin/bash
#SBATCH -p serial 
# Set number of nodes to run
#SBATCH --nodes=1
# Set number of tasks to run
#SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=12 
# memory
#SBATCH --mem=10GB
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


export INPUT_FILE=/scratch/ba63/gender_bias/data/OpenSubtitles.ar-en.ar
export OUTPUT_FILE=/scratch/ba63/gender_bias/data/OpenSubtitles.ar-en.ar.sep.punc

python punc_sep_data.py --input_file_dir $INPUT_FILE --output_file_dir $OUTPUT_FILE
