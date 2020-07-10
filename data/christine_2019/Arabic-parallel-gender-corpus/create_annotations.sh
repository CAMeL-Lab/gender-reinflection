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

# TODO: find a better way to provide inputs and ouputs to this script

module purge
eval "$(conda shell.bash hook)"
conda activate python2

python  ~/m2scorer/scripts/edit_creator.py D-set-dev.ar.F.normalized D-set-dev.ar.M.normalized > edits_annotations_normalized/D-set-dev.F.to.M.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-dev.ar.M.normalized D-set-dev.ar.F.normalized > edits_annotations_normalized/D-set-dev.M.to.F.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-dev.arin.normalized D-set-dev.ar.M.normalized > edits_annotations_normalized/D-set-dev.arin.to.M.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-dev.arin.normalized D-set-dev.ar.F.normalized > edits_annotations_normalized/D-set-dev.arin.to.F.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-train.ar.F.normalized D-set-train.ar.M.normalized > edits_annotations_normalized/D-set-train.F.to.M.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-train.ar.M.normalized D-set-train.ar.F.normalized > edits_annotations_normalized/D-set-train.M.to.F.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-train.arin.normalized D-set-train.ar.M.normalized > edits_annotations_normalized/D-set-train.arin.to.M.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-train.arin.normalized D-set-train.ar.F.normalized > edits_annotations_normalized/D-set-train.arin.to.F.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-test.ar.F.normalized D-set-test.ar.M.normalized > edits_annotations_normalized/D-set-test.F.to.M.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-test.ar.M.normalized D-set-test.ar.F.normalized > edits_annotations_normalized/D-set-test.M.to.F.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-test.arin.normalized D-set-test.ar.M.normalized > edits_annotations_normalized/D-set-test.arin.to.M.edits_annotation.normalized
python  ~/m2scorer/scripts/edit_creator.py D-set-test.arin.normalized D-set-test.ar.F.normalized > edits_annotations_normalized/D-set-test.arin.to.F.edits_annotation.normalized
