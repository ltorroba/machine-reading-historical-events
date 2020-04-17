#!/bin/bash
#
# This script automatically runs 
#

EXPERIMENT_FOLDER=models/simple
EXPERIMENT_FILE=experiment-concat-embeddings.jsonnet
RESULTS_FOLDER=.

#! Full path to application executable:
application="python -m allennlp.run evaluate"

SEED=( 1 2 3 4 5 6 )

for seed in "${SEED[@]}"
do
  EXPERIMENT_SPECIFIC_NAME=run-${seed}
  EXPERIMENT_SET_FOLDER=simple/final/${dataset}/use-ci-${use_ci}/${predictor_type}
  EXPERIMENT_NAME=${EXPERIMENT_SET_FOLDER}/${EXPERIMENT_SPECIFIC_NAME}
  CMD="$application ${RESULTS_FOLDER}/results/${EXPERIMENT_NAME}/model.tar.gz ../data/test/${TESTSET_FILENAME} --include-package models --output-file ${RESULTS_FOLDER}/results/${EXPERIMENT_NAME}/test-metrics.txt"
  echo $EXPERIMENT_NAME
  echo $CMD
  eval $CMD
done
