#!/bin/bash

EXPERIMENT_FOLDER=models/simple
EXPERIMENT_FILE=experiment-concat-embeddings.jsonnet
RESULTS_FOLDER=.

#! Full path to application executable:
application="python -m allennlp.run predict"

#! Run options for the application:
options="--use-dataset-reader --include-package models --predictor dataset-regressor-predictor"

SEED=( 1 2 3 4 5 6 )

#python -m allennlp.run predict results/foo/run-1/model.tar.gz data/validation/or-processed.pkl --use-dataset-reader --include-package models --predictor dataset-regressor-predictor
for seed in "${SEED[@]}"
do
  EXPERIMENT_SPECIFIC_NAME=run-${seed}
  EXPERIMENT_SET_FOLDER=simple/final/${dataset}/use-ci-${use_ci}/${predictor_type}
  EXPERIMENT_NAME=${EXPERIMENT_SET_FOLDER}/${EXPERIMENT_SPECIFIC_NAME}

  CMD="$application ${RESULTS_FOLDER}/results/${EXPERIMENT_NAME}/model.tar.gz ../data/validation/${VALIDATIONSET_FILENAME} --output-file ${RESULTS_FOLDER}/results/${EXPERIMENT_NAME}/validation-set-outputs.txt $options"
  echo $EXPERIMENT_NAME
  echo $CMD
  eval $CMD
done
