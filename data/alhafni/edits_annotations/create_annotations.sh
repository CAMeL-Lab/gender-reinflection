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

python  ~/m2scorer/scripts/edit_creator.py D-set-dev.ar.F D-set-dev.ar.M > edits_annotations/D-set-dev.F.to.M.edits_annotationa
python  ~/m2scorer/scripts/edit_creator.py D-set-dev.ar.M D-set-dev.ar.F > edits_annotations/D-set dev.M.to.F.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-dev.arin D-set-dev.ar.M > edits_annotations/D-set-dev.arin.to.M.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-dev.arin D-set-dev.ar.F > edits_annotations/D-set-dev.arin.to.F.edits_annotation

python  ~/m2scorer/scripts/edit_creator.py D-set-train.ar.F D-set-train.ar.M > edits_annotations/D-set-train.F.to.M.edits_annotati
on
python  ~/m2scorer/scripts/edit_creator.py D-set-train.ar.M D-set-train.ar.F > edits_annotations/D-set-train.M.to.F.edits_annotati
on
python  ~/m2scorer/scripts/edit_creator.py D-set-train.arin D-set-train.ar.M > edits_annotations/D-set-train.arin.to.M.edits_annot
ation
python  ~/m2scorer/scripts/edit_creator.py D-set-train.arin D-set-train.ar.F > edits_annotations/D-set-train.arin.to.F.edits_annot
ation

python  ~/m2scorer/scripts/edit_creator.py D-set-test.ar.F D-set-test.ar.M > edits_annotations/D-set-test.F.to.M.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-test.ar.M D-set-test.ar.F > edits_annotations/D-set-test.M.to.F.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-test.arin D-set-test.ar.M > edits_annotations/D-set-test.arin.to.M.edits_annotati
on
python  ~/m2scorer/scripts/edit_creator.py D-set-test.arin D-set-test.ar.F > edits_annotations/D-set-test.arin.to.F.edits_annotati
on
