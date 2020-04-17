# LSTM Model

This folder contains the code for the LSTM model proposed for the task of historical event ordering.

## Folder Structure

This directory is setup as follows:

1. `models`: Contains logic for all models (only `simple` right now), alongside scripts to handle the `dataset` loader, a `predictor` and `metrics`.
2. `embeddings`: (See installation instructions below) This folder should contain the required Glove and FastText embeddings.
3. `scripts`: Contains the scripts required to reproduce our LSTM experiments.

## Installation

### Packages
In order to run the project, you need to install a few packages. We recommend using [conda](https://docs.conda.io/en/latest/),
and running:

1. `conda create --name heo-lstm python=3.6 pandas`
2. `conda activate heo-lstm`
3. `conda install -c conda-forge jsonnet`
4. Install the appropriate [pyTorch](https://pytorch.org/) package.
5. `pip install allennlp==0.9.0`

### Embeddings

The LSTM model uses pre-trained [200d-Glove](http://nlp.stanford.edu/data/glove.6B.zip) and
[300d-fastText](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip) embeddings.
These should be downloaded, extracted and placed in an `embeddings` folder, under `glove` and `fasttext` folders, respectively.

In the end, the `embeddings` folder should look like this:

```
embeddings/
├── fasttext
│   └── wiki-news-300d-1M-subword.vec
└── glove
    └── glove.6B.200d.txt
```

## Running Experiments

The `scripts` directory contains the scripts required to reproduce the LSTM experiments.
In general, those script will declare environment variables and call a `*.prototype.*` file with
the experiment logic.
This in turn will call the AllenNLP process, which loads a `models/simple/experiment-concat-embeddings.jsonnet` script
that builds the AllenNLP experiment configuration.

Most files are named using the format `{experiment_type}.{dataset}.{ci_use}.{prediction_type}.glove-fasttext`.
The different values of `{experiment_type}` are
- `hyperparam`: Hyperparameter tuning experiments, performing a grid search over various values. Will create a `best-settings.txt` file that orders each hyperparameter combination by its Kendall tau performance on the validation set. Note that hyperparameter tuning experiments run on a subset of the data--this is controlled in the configuration files by enabling setting the environment
variable `hyperparam_mode=true`.
- `final`: Trains multiple models (using different seeds) using the specified hyperparameter values. Each run gets its own directory.
- `predict`: Generates predictions for the validation set using final models, and puts them in a `validation-set-outputs.txt` file in each run directory.
- `test`: Evaluates the final trained models on the test set, and created a `test-metrics.txt` file in each run directory.

There are two additional scripts, `studyA.reg.glove-fasttext` and `studyB.reg.glove-fasttext`, corresponding
to ablation studies A and B on the paper, that use the WOTD dataset.

In order to run the experiments, ensure that you run the script from the `lstm` directory.
For example: `./scripts/final.otd.ci.cls.glove-fasttext` will train 6 models (on different, pre-specified seeds)
on the OTD dataset, using contextual information, with a year classifier.

By default, all results are saved to `lstm/results`, but this can be changed by modifying the `*.prototype.*` files in
`models/simple`, which contain the actual code to run the experiments.

## Other

One way to debug is to use pyCharm, and attach to the AllenNLP process as explained [here](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/using_a_debugger.md#how-to-debug-in-pycharm-using-run--attach-to-local-process).