### Arabic Parallel Gender Corpus:

Here we describe the data we used to train and test our gender reinflection models. <br/>

We have 3 types of files: <br/>

1. The actual data files: **D-set-[train|dev|test].[arin|ar][.M|.F]**. These should the same data files that were obtained from the Arabic Parallel Gender Corpus (Habash et al. 2019)
2. The labels (B, M, or F) files: **D-set-[train|dev|test].[arin|ar][.M|.F].label**. These should also be the same as the labels obtained from the Arabic Parallel Gender Corpus (Habash et al. 2019)
3. The gender (M or F) files: **D-set-[train|dev|test].ar.[M|F].gender**. For the masculine data, the gender file will always have **M**, whereas for the feminine data, the gender file will always have **F**. So for example, **D-set-[train|dev|test].ar.M.gender** will always have the **M** label.

We also create a normalized version of the data files (**D-set-[train|dev|test].[arin|ar][.M|.F].normalized**); we do AYT normaliztion. The normalized files will also have labels and gender files, which are going to be same as the unnormalized gender and label files.

All the data described so far should have the same exact splits and examples, as described by Habash et al. 2019:


| Split    | Num Examples  | 
| -------  |:------:|
| Train | 8566   | 
| Dev | 1224   |   
| Test  | 2448   |
| **Total**| **12238**  |


### Arabic Parallel Gender Corpus (Alhafni):

To train and test our models, we duplicated Corpus<sub>input</sub> (D-set-[train|dev|test].arin) to represent our source corpus. For the target corpus, we concatenating Corpus<sub>M</sub> (D-set-[train|dev|test].ar.M) and Corpus<sub>F</sub> (D-set-[train|dev|test].ar.F). We did the concatention across all splits defined by Habash et al. 2019. We did the same process for the labels and gender files and we also have a normalized version of the data. </br>
This data can be found in `joint_model/D-set-[train|dev|test].[arin|ar][.M|.F]+D-set-[train|dev|test].[arin|ar][.M|.F][.normalized][.gender|.label]`.

After the concatenation, we end up with the following:

| Split    | Num Examples  | 
| -------  |:------:|
| Train | 17132   | 
| Dev | 2448   |   
| Test  | 4896   |
| **Total**| **24476**  |
