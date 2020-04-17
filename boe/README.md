# Bag of Embeddings (BOE) Model

This folder contains the code for the BOE model proposed for the task of historical event ordering.

## Installation

### Packages
In order to run the project, you need to install a few packages.

1. Make Sure you have Python 3.7 installed.
2. Make sure you have the following packages installed: Numpy version 1.16.x, 
scikit-learn version 0.22.x, SciPy version 1.1.x, pandas version 1.0.x.
2. Install spaCy vesrion 2.x and download the 'en_core_web_lg' model. (https://spacy.io/usage)
3. Install [pyTorch](https://pytorch.org/).

### Embeddings

The BOE model uses pre-trained [300d-Glove](http://nlp.stanford.edu/data/glove.6B.zip) embeddings.
These should be downloaded and extracted to your repository.

## Running Experiments

1. Train a classifier and run the experiments:
`python experiments.py`

2. Run ablation study:
`python ablate.py`