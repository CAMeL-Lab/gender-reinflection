# Gender Identification:

This directory has all the logs and gender identification results for all the experiment we report in the paper. </br></br>

`do_nothing/`: the do nothing baseline gender identification results on the dev and test sets.</br>
`mle/`: the bigram MLE baseline gender identification results on the dev and test sets.</br>
`joint_models/`: the joint models gender identification results on the dev set. We also include the results of the best model on the test set.</br>
`disjoint_models/arin.to.[F|M]`: the gender identification results for the masculine and feminine disjoint models on the dev set. `disjoint_models/dev.disjoint+morph.log` has gender identification results for the merged masculine and feminine disjoint models ouputs on the dev set </br>

To access the results, just do `tail -22 *.log`.
