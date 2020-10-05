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

export DATA_DIR=data/alhafni
 
#--inference_data logs/reinflection/joint_models/test.joint+morph.inf.norm
python utils/gender_identification.py \
 --data_dir $DATA_DIR \
 --normalized \
 --inference_mode test \
 --inference_data habash_2019/test.arin.to.M+F.norm

#/Users/ba63/Desktop/repos/gender-bias/habash_2019/gender_id_results/hypotheses/dev.arin.to.M+F.norm

#/Users/ba63/Desktop/repos/gender-bias/logs/reinflection/disjoint_models/dev.arin.to.M+F.disjoint+morph.inf.norm
