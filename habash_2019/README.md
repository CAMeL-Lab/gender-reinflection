## [Habash et. al 2019](https://www.aclweb.org/anthology/W19-3822.pdf) Evaluation:


`dev.arin.to.M+F.norm`: is the concatenatation of `char-level-NMT/MT-output/select/arin/selection-output-dev-arin-MTarget.txt.normh` and `char-level-NMT/MT-output/select/arin/selection-output-dev-arin-FTarget.txt.normh` (i.e. `cat char-level-NMT/MT-output/select/arin/selection-output-dev-arin-MTarget.txt.normh char-level-NMT/MT-output/select/arin/selection-output-dev-arin-FTarget.txt.normh > dev.arin.to.M+F.norm`)

`test.arin.to.M+F.norm`: is the concatenatation of `char-level-NMT/MT-output/select/arin/selection-output-test-arin-MTarget.txt.normh` and `char-level-NMT/MT-output/select/arin/selection-output-test-arin-FTarget.txt.normh` (i.e. `cat char-level-NMT/MT-output/select/arin/selection-output-test-arin-MTarget.txt.normh char-level-NMT/MT-output/select/arin/selection-output-test-arin-FTarget.txt.normh > test.arin.to.M+F.norm`)

To run the reinflection evaluation, you would need to run `run_reinflection_eval.sh`. Make sure to change the `$DATA_SPLIT` from dev to test to run the evaluation on the dev and the test data respectively. 

To run the gender identification evaluation, you would need to run `gender_identification.sh`. Make sure to change the `inference_mode` from dev to test to run the evaluation on the dev and the test data respectively.

`gender_id_results` contain the gender identification logs on both the dev and test splits. `reinflection_results` contains the reinflection evaluation on both the dev and test splits. 
