#!/bin/bash

# Evaluate all models
# Usage: ./evaluate_all_models.sh
conda activate egises_sub_evaluation
# take first argument in score default to perseval
score=${1:-"perseval"}
version=${2:-"v2"}
simplied_flag=${3:-"False"}

cd ..
# loop over all models
for measure in "bleu" "rougeL" "rougeSU4" "meteor" "infoLM" "bert_score" "JSD"
do
    # evaluate model
    if [ $score == "perseval" ]
    then
      if [ $simplied_flag == "True" ]
        then
        python evaluation_script.py generate-perseval-scores $measure --simplified-flag --version $version
      else
        python evaluation_script.py generate-perseval-scores $measure --no-simplified-flag --version $version
      fi
    else
      if [ $simplied_flag == "True" ]
        then
        python evaluation_script.py generate-scores $measure --simplified-flag --version $version
      else
        python evaluation_script.py generate-scores $measure --no-simplified-flag --version $version
      fi
    fi
done
