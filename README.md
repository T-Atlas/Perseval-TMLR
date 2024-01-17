# Egises subjectivity experiments
This repository contains the code for the experiments on subjectivity detection in the EGISES project.

[//]: # (PERSONALIZED_MODELS = &#40;"NAML_1", "NRMS_1", "NRMS_2", "EBNR_1", "EBNR_2"&#41;)
[//]: # (NON_PERSONALIZED_MODELS_LIST = &#40;"big_bird", "brio", "prophetnet", "cls", "t5_base"&#41;)
## Models
- Personalized models: `NAML_1`, `NRMS_1`, `NRMS_2`, `EBNR_1`, `EBNR_2`
- Non-personalized models: `big_bird`, `brio`, `prophetnet`, `cls`, `t5_base`

## Measures
- rougeL
- rougeSU4
- meteor
- bleu
- bertscore
- infoLM (pending)
## Scripts
- `data_preprocessing.py`: 
  - consolidates previously generate data into a single jsonl file  
  - tokenize the text in consolidated file and save it in a new file
  - test functions to verify consolidation and tokenization
  - export hj scores from database file
    ```
    # human judgements are scored in a sqlite file. They have summary-summary pairs(via user ratings), but not summary-doc, user-doc, summary-user(accuracy) scores. So we use auxilary distance measures to fill that gap
    python data_preprocessing.py export-hj-data-to-csv --help
    │ --database-path                 TEXT  [default: dataset/survey_db_v3.sqlite3]               │
    │ --distance-path                 TEXT  [default: scores/calculate_hj]                        │
    │ --auxilary-distance-path        TEXT  [default: scores/calculate_rougL]                     │
    │ --help                                Show this message and exit.
    ```
  - For more details:
    ```bash
        python data_preprocessing.py --help
    ```
- `evaluation_script.py`: 
  - Calculate distance measures between the different models.
    - user summary distances from document distances
    - model summary distances from document distances
    - user-model summary distances from each other
  - Calculate the correlation between egises-measure(measure plugged in egises) scores and measure
  - For more details:
    ```bash
        python evaluation_script.py --help
         Commands ──────────────────────────────────────────────────────────────────────────────────╮
        │ calculate-correlation     dmeasure_1: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd    │
        │                           dmeasure_2: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd,   │
        │                           hj p_measure: one of egises, perseval                             │
        │ generate-perseval-scores  distance_measure: one of meteor, bleu, rougeL, rougeSU4, infoLM,  │
        │                           jsd model_name: sampling frequency for percentage less than 100   │
        │                           max_workers: number of workers to use for multiprocessing         │
        │                           version: generate suffixed scores files to avoid overwriting      │
        │                           saves scores in                                                   │
        │                           scores/distance_measure/perseval_scores_version.csv               │
        │ generate-scores           distance_measure: one of meteor, bleu, rougeL, rougeSU4, infoLM,  │
        │                           jsd sampling_freq: sampling frequency for percentage less than    │
        │                           100 max_workers: number of workers to use for multiprocessing     │
        │                           version: generate suffixed scores files to avoid overwriting      │
        │                           saves scores in scores/distance_measure/egises_scores_version.csv │
        │ populate-distances        model_name: one of PERSONALIZED_MODELS or                         │
        │                           NON_PERSONALIZED_MODELS_LIST distance_measure: one of meteor,     │
        │                           bleu, rougeL, rougeSU4, infoLM, jsd max_workers: number of        │
        │                           workers to use for multiprocessing                                │
        ─────────────────────────────────────────────────────────────────────────────────────────────
    ```

- `bash scripts`
  - run python scripts for different models, measures

For a quick walkthrough of the functions, please refer to `evaluation_notebook.ipynb`
