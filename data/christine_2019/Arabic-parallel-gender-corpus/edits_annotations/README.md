This folder has the edits annotations which were created by the m2scorer edit creator script.

python  ~/m2scorer/scripts/edit_creator.py D-set-dev.ar.F D-set-dev.ar.M > edits_annotations/D-set-dev.F.to.M.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-dev.ar.M D-set-dev.ar.F > edits_annotations/D-set dev.M.to.F.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-dev.arin D-set-dev.ar.M > edits_annotations/D-set-dev.arin.to.M.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-dev.arin D-set-dev.ar.F > edits_annotations/D-set-dev.arin.to.F.edits_annotation

python  ~/m2scorer/scripts/edit_creator.py D-set-train.ar.F D-set-train.ar.M > edits_annotations/D-set-train.F.to.M.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-train.ar.M D-set-train.ar.F > edits_annotations/D-set-train.M.to.F.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-train.arin D-set-train.ar.M > edits_annotations/D-set-train.arin.to.M.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-train.arin D-set-train.ar.F > edits_annotations/D-set-train.arin.to.F.edits_annotation

python  ~/m2scorer/scripts/edit_creator.py D-set-test.ar.F D-set-test.ar.M > edits_annotations/D-set-test.F.to.M.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-test.ar.M D-set-test.ar.F > edits_annotations/D-set-test.M.to.F.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-test.arin D-set-test.ar.M > edits_annotations/D-set-test.arin.to.M.edits_annotation
python  ~/m2scorer/scripts/edit_creator.py D-set-test.arin D-set-test.ar.F > edits_annotations/D-set-test.arin.to.F.edits_annotation
