# Linguistically Enhanced Neural Models for Joint Gender Identification and Reinflection:
This repo contains code to reproduce the results in our paper [Linguistically Enhanced Neural Models for Joint Gender Identification and Reinflection](blabla)

## Requirements:
The code was written for python>=3.6 and pytorch 1.3, although newer versions of pytorch might work just fine. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda installed):

TODO: create the setup.py file
```
git clone https://github.com/balhafni/gender-bias.git
cd gender-bias

conda create -n python3 python=3.6
conda activate python3

pip install .
```

## Training the model:

To train the best joint reinflection and identification (`joint+morph`) model we describe in our paper, you need to run `sbatch scripts/train_seq2seq.sh`. Training the model should take around 3 hours on a single GPU, although this may vary based on the GPU you're using. Once the training is done, the trained pytorch model will be saved in `saved_models/`.

## Evaluation:



## Gender Identification:

## Error Analysis:
