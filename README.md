# Gender-Aware Reinflection using Linguistically Enhanced Neural Models:
This repo contains code to reproduce the results in our paper [Gender-Aware Reinflection using Linguistically Enhanced Neural Models](blabla)

## Requirements:
The code was written for python>=3.6 and pytorch 1.3, although newer versions of pytorch might work just fine. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda installed):

```bash
git clone https://github.com/balhafni/gender-reinflection.git
cd gender-reinflection

conda create -n gender_reinflection python=3.6
conda activate gender_reinflection

pip install .
```

## Training the model:

To train the best joint reinflection and identification (`joint+morph`) model we describe in our paper, you need to run `sbatch scripts/train_seq2seq.sh`. Training the model should take around 3 hours on a single GPU, although this may vary based on the GPU you're using. Once the training is done, the trained pytorch model will be saved in `saved_models/`.

## Inference:

To get the gender reinflected sentences based on the seq2seq model, you would need to run `sbatch scripts/inference_seq2seq.sh`. The inference script will produce 3 files: .beam (beam search with beam size=10), .inf (greedy search), and .beam_greedy (beam search with beam size=1, i.e. greedy search).</br>
To get the gender reinflected sentences based on the bigram MLE model, you would need to run `sbatch scripts/mle_inference.sh`. </br></br>

Refer to [logs/reinflection](https://github.com/balhafni/gender-bias/tree/master/logs/reinflection) to get the reinflected sentences for all the experiments we report on in our paper.

## Evaluation:

We use the M<sup>2</sup> scorer and SacreBLEU in our evaluation. To run the evaluation, for the MLE, do nothing, and joint models we report on in the paper, you would need to run `sbatch scripts/run_eval_norm_joint.sh`. Make sure to change the path of inference data you want to evaluate on (refer to `SYSTEM_HYP` in ` scripts/run_eval_norm_joint.sh`). </br></br>
To run the evaluation for the disjoint models, you would need to run `sbatch scripts/run_eval_norm_disjoint.sh`. Note that we merge the masculine disjoint system output (`logs/reinflection/disjoint_models/arin.to.M/dev.disjoint+morph.inf.norm`) and the feminine disjoint system output (`logs/reinflection/disjoint_models/arin.to.F/dev.disjoint+morph.inf.norm`) and we evaluate on the merged output (`logs/reinflection/disjoint_models/dev.disjoint+morph.inf.norm`). This is like reporting the average of both systems together. </br></br>


## Gender Identification:

To get the results of gender identification we report for our experiments in the paper, you would need to run `sbatch scripts/gender_identification.sh`. Make sure to change the inference data path based on the experiment you're running. Throughout all experiments, we report the average F<sub>1</sub> score over the masculine and feminine data. </br></br>
Refer to [logs/gender_id](https://github.com/balhafni/gender-bias/tree/master/logs/gender_id) to get the gender id logs based on how we defined gender identification in our paper.

## Error Analysis:

We also conduct a simple error analysis to indicate which words changed during inference. This helped us in conducting a more thourough manual error analysis which we reported in the paper. We did the error analysis on the results of our best model (`joint+morph`) on the dev set on the feminine and masculine data separately. To run the error analysis script, you would need to run `sbatch scripts/error_analysis`. Make sure to change the `EXPERIMENT_NAME` to `arin.to.F` to run the error analysis over the feminie dev set results and to `arin.to.M` to run the error analysis over the masuline dev set.  </br></br>

Refer to [logs/error_analysis](https://github.com/balhafni/gender-bias/tree/master/logs/error_analysis) to get the error analysis logs.
