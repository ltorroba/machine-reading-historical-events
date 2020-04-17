#!/bin/bash

EXPERIMENT_FOLDER=models/simple
EXPERIMENT_FILE=experiment-concat-embeddings.jsonnet
RESULTS_FOLDER=.

#! Full path to application executable:
application="python -m allennlp.run train"

#! Run options for the application:
options="${EXPERIMENT_FOLDER}/${EXPERIMENT_FILE} --include-package models"

ENCODER_NUM_LAYERS=( 1 2 3 )
ENCODER_HIDDEN_DIM=( 50 100 200 300 )
PREDICTOR_NUM_LAYERS=( 1 2 3 )
PREDICTOR_HIDDEN_DIM=( 50 100 200 300 )

for encoder_num_layers in "${ENCODER_NUM_LAYERS[@]}"
do
  export encoder_num_layers
  for encoder_hidden_dim in "${ENCODER_HIDDEN_DIM[@]}"
  do
    export encoder_hidden_dim
    for predictor_num_layers in "${PREDICTOR_NUM_LAYERS[@]}"
    do
      export predictor_num_layers
      for predictor_hidden_dim in "${PREDICTOR_HIDDEN_DIM[@]}"
      do
        export predictor_hidden_dim
        EXPERIMENT_SPECIFIC_NAME=EL${encoder_num_layers}-ED${encoder_hidden_dim}-PL${predictor_num_layers}-PD${predictor_hidden_dim}
        EXPERIMENT_SET_FOLDER=simple/hyperparam/${dataset}/use-ci-${use_ci}/${predictor_type}
        EXPERIMENT_NAME=${EXPERIMENT_SET_FOLDER}/${EXPERIMENT_SPECIFIC_NAME}
        CMD="$application $options -s ${RESULTS_FOLDER}/results/${EXPERIMENT_NAME}"
        echo $EXPERIMENT_NAME
        echo $CMD
        eval $CMD

        # Remove model files, we don't need them during hyperparam tuning, just the metrics
        # This will ensure we don't overfill RDC.
        rm ${RESULTS_FOLDER}/results/${EXPERIMENT_NAME}/*.th
        rm ${RESULTS_FOLDER}/results/${EXPERIMENT_NAME}/model.tar.gz
      done
    done
  done
done

# Sort best hyperparam results
cd ${RESULTS_FOLDER}/results/${EXPERIMENT_SET_FOLDER}
echo -e "Changed directory to `pwd`.\n"

grep -R "best_validation_kendall_tau" **/metrics.json | sed 's/.$//' | sort -b -k3,4 -n > best-settings.txt
