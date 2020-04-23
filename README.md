# Grammatical Error Correction to Reduce Gender Bias in Arabic

### GEC Literature Review:

[Grammatical error correction using neural machine translation, NAACL 2016](https://www.aclweb.org/anthology/N16-1042.pdf):

* The authors presented the first study using neural machine translation (NMT) for grammatical error correction (GEC). The used unsupervised alignments (METEOR) to deal with OOV words as a post-processing step.

[Neural Language Correction with Character-Based Attention, 2016](https://arxiv.org/pdf/1603.09727.pdf):

* NOTE: This is the first paper of GEC as a char-level encoder decoder model.
* The authors use a char-level encoder decoder model with attention for GEC. 

[Better Evaluation for Grammatical Error Correction](https://www.aclweb.org/anthology/N12-1067.pdf):

* NOTE: This is an evaluation metric paper.
* The authors introduce the famous m2 (Max-Match) metric

[Automatic Annotation and Evaluation of Error Types for Grammatical Error Correction, ACL 2017](https://www.aclweb.org/anthology/P17-1074.pdf):

* NOTE: This is an evaluation metric paper.
* The authors present a new mteric ERRANT (ERRor ANnotation Toolkit).
* They claim that the m2scorer evaluation do some sort of over estimating while computing the F0.5 score. 
