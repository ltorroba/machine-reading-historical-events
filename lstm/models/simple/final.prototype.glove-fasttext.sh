#!/bin/bash

EXPERIMENT_FOLDER=models/simple
EXPERIMENT_FILE=experiment-concat-embeddings.jsonnet
RESULTS_FOLDER=.

#! Full path to application executable:
application="python -m allennlp.run train"

#! Run options for the application:
options="${EXPERIMENT_FOLDER}/${EXPERIMENT_FILE} --include-package models"

SEED=( 1 2 3 4 5 6 )

for seed in "${SEED[@]}"
do
  EXPERIMENT_SPECIFIC_NAME=run-${seed}
  EXPERIMENT_SET_FOLDER=simple/final/${dataset}/use-ci-${use_ci}/${predictor_type}
  EXPERIMENT_NAME=${EXPERIMENT_SET_FOLDER}/${EXPERIMENT_SPECIFIC_NAME}
  CMD="$application $options -s ${RESULTS_FOLDER}/results/${EXPERIMENT_NAME} --overrides \"{'random_seed': '${seed}','numpy_seed': '${seed}','pytorch_seed': '${seed}'}\""
  echo $EXPERIMENT_NAME
  echo $CMD
  eval $CMD
done
