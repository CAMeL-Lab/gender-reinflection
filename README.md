# Gender-Aware Reinflection using Linguistically Enhanced Neural Models:
This repo contains code to reproduce the results in our paper [Gender-Aware Reinflection using Linguistically Enhanced Neural Models](https://www.aclweb.org/anthology/2020.gebnlp-1.12.pdf)

## Requirements:
The code was written for python>=3.6 and pytorch 1.3, although newer versions of pytorch might work just fine. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):

```bash
git clone https://github.com/CAMeL-Lab/gender-reinflection.git
cd gender-reinflection

conda create -n gender_reinflection python=3.6
conda activate gender_reinflection

pip install -r requirements.txt
```

## Training the model:

To train the best joint reinflection and identification (`joint+morph`) model we describe in our paper, you need to run `sbatch scripts/train_seq2seq.sh`. Training the model should take around 3 hours on a single GPU, although this may vary based on the GPU you're using. Once the training is done, the trained pytorch model will be saved in `saved_models/`.

## Inference:

To get the gender reinflected sentences based on the trained seq2seq model, you would need to run `sbatch scripts/inference_seq2seq.sh`. The inference script will produce 3 files: .beam (beam search with beam size=10), .inf (greedy search), and .beam_greedy (beam search with beam size=1, i.e. greedy search).</br>
To get the gender reinflected sentences based on the bigram MLE model, you would need to run `sbatch scripts/mle_inference.sh`. </br></br>

Refer to [logs/reinflection](/logs/reinflection) to get the reinflected sentences for all the experiments we report on in our paper.

## Reinflection Evaluation:

We use the M<sup>2</sup> scorer and SacreBLEU in our evaluation. To run the evaluation, for the MLE, do nothing, and joint models we report on in the paper, you would need to run `sbatch scripts/run_eval_norm_joint.sh`. Make sure to change the path of inference data you want to evaluate on (refer to `SYSTEM_HYP` in ` scripts/run_eval_norm_joint.sh`). </br></br>
To run the evaluation for the disjoint models, you would need to run `sbatch scripts/run_eval_norm_disjoint.sh`. Note that we merge the masculine disjoint system output (`logs/reinflection/disjoint_models/arin.to.M/dev.disjoint+morph.inf.norm`) and the feminine disjoint system output (`logs/reinflection/disjoint_models/arin.to.F/dev.disjoint+morph.inf.norm`) and we evaluate on the merged output (`logs/reinflection/disjoint_models/dev.disjoint+morph.inf.norm`). This is the same as reporting the average of both systems. </br></br>


## Gender Identification Evalutation:

To get the results of gender identification we report for our experiments in the paper, you would need to run `sbatch scripts/gender_identification.sh`. Make sure to change the `inference_data` path and the `inference_mode` based on the experiment you're running. Throughout all experiments, we report the average F<sub>1</sub> score over the masculine and feminine data. </br></br>
Refer to [logs/gender_id](/logs/gender_id) to get the gender id logs based on how we defined gender identification in our paper.

## Error Analysis:

We also conduct a simple error analysis to indicate which words changed during inference. This helped us in conducting a more thourough manual error analysis which we reported in the paper. We did the error analysis on the results of our best model (`joint+morph`) on the dev set on the feminine and masculine data separately. To run the error analysis script, you would need to run `sbatch scripts/error_analysis`. Make sure to change the `EXPERIMENT_NAME` to `arin.to.F` to run the error analysis over the feminie dev set results and to `arin.to.M` to run the error analysis over the masuline dev set.  </br></br>

Refer to [logs/error_analysis](/logs/error_analysis) to get the error analysis logs.

## License:

This repo is available under the MIT license. See the [LICENSE file](/LICENSE) for more info.

## Citation:

If you find the code or data in this repo helpful, please cite [our paper](https://www.aclweb.org/anthology/2020.gebnlp-1.12.pdf):

```bibtex
@inproceedings{alhafni-etal-2020-gender,
    title = "Gender-Aware Reinflection using Linguistically Enhanced Neural Models",
    author = "Alhafni, Bashar  and
      Habash, Nizar  and
      Bouamor, Houda",
    booktitle = "Proceedings of the Second Workshop on Gender Bias in Natural Language Processing",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.gebnlp-1.12",
    pages = "139--150",
    abstract = "In this paper, we present an approach for sentence-level gender reinflection using linguistically enhanced sequence-to-sequence models. Our system takes an Arabic sentence and a given target gender as input and generates a gender-reinflected sentence based on the target gender. We formulate the problem as a user-aware grammatical error correction task and build an encoder-decoder architecture to jointly model reinflection for both masculine and feminine grammatical genders. We also show that adding linguistic features to our model leads to better reinflection results. The results on a blind test set using our best system show improvements over previous work, with a 3.6{\%} absolute increase in M2 F0.5.",
}
