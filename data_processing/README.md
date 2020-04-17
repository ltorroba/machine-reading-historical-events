# data_processing Model

This folder contains the code for CI extraction. 
Note that the datasets stored in the `data` folder already contain the extracted CI, so it is
not necessary to use this model in order to run the LSTM and BOE experiments.

## Installation

### Packages
In order to run the project, you need to install a few packages.

1. Make Sure you have Python 3.7 installed.
2. Make sure you have the following packages installed:
Numpy version 1.16.x, pandas version 1.0.x.
3. Install Wikipedia API for Python. (https://pypi.org/project/wikipedia)
4. Install spaCy vesrion 2.x and download the 'en_core_web_lg' model. (https://spacy.io/usage)


## Running Extraction Process
In order to extract CI for a historical events, run:
`python extract_ci in_file out_file`
When `in_file` is the input pickle file that contains only events descriptions and 
years, and `out_file` is the name for the output pickle file.
