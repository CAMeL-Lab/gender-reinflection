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
 
# To run the gender ID evaluation on the joint model's output,
# you have to provide the output of the joint model's at
# logs/reinflection/joint_models/[test|dev].joint+morph.inf.norm

# To run the gender ID evaluation on the disjoint model's output,
# you have to provide the output of the disjoint model's at
# logs/reinflection/disjoint_models/dev.disjoint+morph.inf.norm
# dev.disjoint+morph.inf.norm is the concatenation of the masculine
# feminine disjoint systems outputs on the dev set

python utils/gender_identification.py \
 --data_dir $DATA_DIR \
 --normalized \
 --inference_mode test \
 --inference_data logs/reinflection/joint_models/test.joint+morph.inf.norm
