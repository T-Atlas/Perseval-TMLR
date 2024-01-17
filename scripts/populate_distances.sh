#!/bin/bash

# Evaluate all models
# Usage: ./evaluate_all_models.sh
#PERSONALIZED_MODELS = ("NAML_1", "NRMS_1", "NRMS_2", "EBNR_1", "EBNR_2")
#NON_PERSONALIZED_MODELS_LIST = ("big_bird", "brio", "prophetnet", "cls", "t5_base")
conda activate egises_sub_evaluation
cd ..
# loop over all models
for model in "NAML_1" "NRMS_1" "NRMS_2" "EBNR_1" "EBNR_2" "big_bird" "brio" "prophetnet" "cls" "t5_base"
# loop over all measures (for each model)
  do
    for measure in "infoLM" #"rougeL" "rougeSU4" "meteor"
      do
          # evaluate model
          python3 -u  evaluation_script.py populate-distances $model $measure
      done
  done