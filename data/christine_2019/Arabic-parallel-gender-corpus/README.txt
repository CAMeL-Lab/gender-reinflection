================================================================

                The Arabic Parallel Gender Corpus

                            Release #1
                          6 August 2019

=================================================================
Summary
=================================================================

The Arabic Parallel Gender Corpus is a corpus designed to support
research on gender bias in natural language processing applications
working on Arabic.  The corpus includes multiple parts with different
features.  In this release, we share the data presented in the 2019
paper on "Automatic Gender Identification and Reinflection in Arabic"
by Habash et al. in the First workshop on Gender Bias in Natural
Language Processing.  This includes Part D (Open Subtitles data with
parallel gender versions in Arabic) and Part S (Synthetic parallel Arabic
gender corpus).  The Open Subtitles data comes from (Lison and Tiedemann,
2016).

When citing this resource, please use:

Habash, Nizar, Houda Bouamor, Christine Chung. 2019. Automatic Gender
Identification and Reinflection in Arabic. In Proceedings of the First
Workshop on Gender Bias in Natural Language Processing, Florence, Italy.

@inproceedings{habash-etal-2019-automatic,
    title = {Automatic Gender Identification and Reinflection in {A}rabic},
    author = {Habash, Nizar and Bouamor, Houda  and Chung, Christine},
    booktitle = {Proceedings of the First Workshop on Gender Bias in Natural Language Processing},
    year = {2019},
    address = {Florence, Italy},
    url = "https://www.aclweb.org/anthology/W19-3822",
    pages = "155--165"
}

=================================================================
Description of Data
=================================================================
The zipped folder "Arabic-parallel-gender-corpus.zip" has the following
contents:

README.txt  :   This file.

LICENSE.txt :   The license to use this corpus.

2019-GenderBiasNLP-Habash-Bouamor-Chung-Paper.pdf :
                A paper describing the creation and use of the corpus.

D-set-*     :   24 files representing the Part D of the corpus.

                There are 8 files for each of the train, dev and test splits:
                D-set-train.*, D-set-dev.*, and D-set-test.*, respectively.
                The corpus provided is the Balanced Corpus presented in the
                above mentioned paper.  The 8 files have unique extensions:

                *.arin              The Arabic input text
                *.en                The English text

                *.ar.F              The Feminine Target text
                *.ar.M              The Masculine Target text

                *.arin.mada.feats3  The morphological features used for the
                                      input text

                *.arin.label        The B/F/M label of the input text
                *.ar.F.label        The B/F label of the Feminine Target text
                *.ar.M.label        The B/M label of the Masculine Target text

S-set.*    :    Two files (S-set.F and S-set.M) containing the feminine and
                masculine forms of the Sybthetic Arabic parallel gender corpus.

================================================================
References
================================================================
Habash, Nizar, Houda Bouamor, Christine Chung. 2019. Automatic Gender
Identification and Reinflection in Arabic. In Proceedings of the First
Workshop on Gender Bias in Natural Language Processing, Florence, Italy.

Lison, Pierre and Jörg Tiedemann. 2016. OpenSubtitles2016: Extracting Large
Parallel Corpora from Movie and TV Subtitles. In Proceedings of the Language
Resources and Evaluation Conference (LREC), Portorož, Slovenia.

================================================================
Copyright (c) 2019 New York University Abu Dhabi. All rights reserved.
================================================================
