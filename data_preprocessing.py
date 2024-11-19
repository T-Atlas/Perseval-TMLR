import os
import random
import shutil
import sqlite3

import pandas as pd
import typer
import ujson
from tqdm.auto import tqdm
import utils

app = typer.Typer()

DATA_SET_PATH = "dataset"
SCORES_PATH = f"scores"

PERSONALIZED_MODELS = ("NAML_1", "NRMS_1", "NRMS_2", "EBNR_1", "EBNR_2")
NON_PERSONALIZED_MODELS_LIST = ("big_bird", "brio", "prophetnet", "cls", "t5_base")


@app.command()
def consolidate_experiment_data(consolidated_filename: str = "consolidated_data.jsonl", dataset_path=DATA_SET_PATH):
    """
    consolidated_filepath: path to consolidated_data.jsonl
    expects all model files in DATA_SET_PATH
    generates combined file in consolidated_data.jsonl
    """
    org_file = f"{dataset_path}/org_dataset/combined_raw_dataset"
    org_dict = utils.load_data(org_file)

    consolidated_filepath = f"{dataset_path}/{consolidated_filename}"

    # get personalized models data
    if os.path.exists(consolidated_filepath):
        print(f"file {consolidated_filepath} already exists. To regenerate results delete it and rerun function.")
        return

    with open(consolidated_filepath, "w") as fp:
        for doc_id, doc_obj in tqdm(org_dict.items()):
            new_doc = {"doc_id": doc_id, "doc_text": doc_obj["doc_text"], "doc_summ": doc_obj["doc_summ"]}
            m_summary_dict = {uid: {"user": usummary} for uid, usummary in doc_obj["u_dict"].items()}
            for model_name, u_object in doc_obj["m_dict"].items():
                for uid, s_text in u_object.items():
                    m_summary_dict[uid][model_name] = s_text
            # m_summary_dict =
            new_doc["m_summary_dict"] = m_summary_dict
            # print(new_doc)
            fp.write(ujson.dumps(new_doc))
            fp.write("\n")

    # load non_personalized_results:

    consolidated_dict = {}
    with open(consolidated_filepath, "r") as fpr:
        for line in tqdm(fpr.readlines()):
            line = line.strip("\n")
            # print(line[-100:])
            line = ujson.loads(line)
            consolidated_dict[line["doc_id"]] = line
        for model_name in tqdm([*NON_PERSONALIZED_MODELS_LIST]):
            source_file = f"{dataset_path}/org_dataset/{model_name}_data"
            source_dict = utils.load_data(source_file)
            for doc_id, doc_obj in source_dict.items():
                for uid, summary_text in doc_obj["m_dict"][model_name].items():
                    try:
                        consolidated_dict[doc_id]["m_summary_dict"][uid][model_name] = summary_text
                    except KeyError as err:
                        print(consolidated_dict[doc_id]["m_summary_dict"])
                        continue

    os.remove(consolidated_filepath)
    with open(consolidated_filepath, "w") as fp:
        for val in consolidated_dict.values():
            fp.write(ujson.dumps(val))
            fp.write("\n")


@app.command()
def test_dataset_consolidation(number_of_documents: int = 10):
    """
    test dataset consolidation script
    """
    org_file = f"{DATA_SET_PATH}/org_dataset/combined_raw_dataset"
    org_dict = utils.load_data(org_file)
    model_name = random.choice(PERSONALIZED_MODELS)

    np_model_name = random.choice(NON_PERSONALIZED_MODELS_LIST)

    np_source_file = f"{DATA_SET_PATH}/org_dataset/{np_model_name}_data"
    np_source_dict = utils.load_data(np_source_file)

    print(f"testing for personalized {model_name}")
    trials = 0
    for document in utils.get_model_documents(model_name):
        if trials >= number_of_documents:
            break
        if random.random() < 0.5:
            continue

        # test document data
        assert org_dict[document.doc_id]["doc_text"] == org_dict[document.doc_id]["doc_text"]
        assert org_dict[document.doc_id]["doc_summ"] == org_dict[document.doc_id]["doc_summ"]

        # test user summary
        for summary in document.user_summaries:
            assert summary.summary_text == org_dict[document.doc_id]["u_dict"][summary.uid]

        for summary in document.model_summaries:
            # print()
            assert summary.summary_text == org_dict[document.doc_id]["m_dict"][model_name][summary.uid]
        trials += 1

    print(f"testing for non personalized {np_model_name}")
    trials = 0
    for document in utils.get_model_documents(np_model_name):
        if trials >= number_of_documents:
            break
        if random.random() < 0.5:
            continue
        # test user summary
        for summary in document.user_summaries:
            assert summary.summary_text == np_source_dict[document.doc_id]["u_dict"][summary.uid]

        for summary in document.model_summaries:
            # print()
            assert summary.summary_text == np_source_dict[document.doc_id]["m_dict"][np_model_name][summary.uid]

        trials += 1
    print(f"tested successfully for {trials} random documents")


@app.command()
def tokenize_experiment_data(input_path: str = f"{DATA_SET_PATH}/consolidated_data.jsonl",
                             output_filepath: str = f"{DATA_SET_PATH}/final_tokenized_consolidated_data.jsonl"):
    """
    input_path: path to consolidated_data.jsonl
    output_filepath: path to final_tokenized_consolidated_data.jsonl
    """

    with open(output_filepath, "w") as fpw:
        with open(input_path, "r") as fp:
            for line in tqdm(fp.readlines()):
                line = line.strip()
                line = ujson.loads(line)
                line["doc_text"] = " ".join(utils._tokenize(line["doc_text"]))
                line["doc_summ"] = " ".join(utils._tokenize(line["doc_summ"]))
                new_summ_dict = {}
                for uid, s_dict in line["m_summary_dict"].items():
                    new_summ_dict[uid] = {}
                    for model, summary in s_dict.items():
                        new_summ_dict[uid][model] = " ".join(utils._tokenize(summary))
                line["m_summary_dict"] = new_summ_dict
                fpw.write(f"{ujson.dumps(line)}\n")


@app.command()
def test_tokenization(filepath: str = f"{DATA_SET_PATH}/consolidated_data.jsonl",
                      tokenized_filepath: str = f"{DATA_SET_PATH}/final_tokenized_consolidated_data.jsonl",
                      test_samples: int = 20):
    """
    filepath: path to consolidated_data.jsonl
    tokenized_filepath: path to final_tokenized_consolidated_data.jsonl
    test_samples: number of samples to test
    """
    trials = 0
    # pick random numbers in 0 to lines in inputfile and output file
    with open(filepath, "r") as fpr, open(tokenized_filepath, "r") as fpr2:
        for line, t_line in zip(fpr.readlines(), fpr2.readlines()):
            if trials >= test_samples:
                break
            if random.random() < 0.5:
                continue

            line = line.strip()
            t_line = t_line.strip()
            line = ujson.loads(line)
            t_line = ujson.loads(t_line)
            assert utils._tokenize_text(line["doc_text"]) == t_line["doc_text"]
            assert utils._tokenize_text(line["doc_summ"]) == t_line["doc_summ"]
            for uid, s_dict in line["m_summary_dict"].items():
                for model, summary in s_dict.items():
                    assert utils._tokenize_text(summary) == t_line["m_summary_dict"][uid][model]
            trials += 1
    print(f"tested successfully for {trials} random documents")


@app.command()
def export_hj_data_to_csv(database_path: str = "dataset/survey_db_v3.sqlite3",
                          distance_path: str = f"{SCORES_PATH}/calculate_hj",
                          auxilary_distance_path: str = f"{SCORES_PATH}/calculate_rougL"):
    """
    database_path: path to database file
    distance_path: path to store distance csv
    auxilary_distance_path: path from where user doc, summary doc distances are copied
    """

    # read sqlite file
    if not os.path.exists(database_path):
        return
    conn = sqlite3.connect(database_path)
    query = f"select doc_id, summ_source as origin_model, summ_id1 as uid1, summ_id2 as uid2, score as rating from response"
    df = pd.read_sql_query(
        query,
        conn)
    min_rating, max_rating = 1, 6
    df["score"] = df.apply(
        lambda row: round(float(1.0 - ((int(row["rating"]) - min_rating) / (max_rating - min_rating))), 5), axis=1)

    df = df.drop(columns=["rating"])
    print(f"before len(df): {len(df)}")
    # take mean of scores for same doc_id, uid1, uid2, origin_model
    df = df.groupby(["doc_id", "uid1", "uid2", "origin_model"]).mean().reset_index()
    # drop redundant columns
    print(f"before len(df): {len(df)}")
    print(df.head())
    origin_models = df["origin_model"].unique()
    for model in origin_models:
        # select rows with origin_model either "user" or model
        if model == "user":
            continue
        model_values = ["user", model]
        model_df = df.loc[df['origin_model'].isin(model_values)]
        if not os.path.exists(f"{distance_path}/{model}"):
            os.makedirs(f"{distance_path}/{model}", exist_ok=True)
        model_df.to_csv(f"{distance_path}/{model}/sum_sum_doc_distances.csv", index=False)
        # copy sum-doc distances
        if os.path.exists(f"{auxilary_distance_path}/{model}/sum_doc_distances.csv"):
            shutil.copyfile(f"{auxilary_distance_path}/{model}/sum_doc_distances.csv",
                            f"{distance_path}/{model}/sum_doc_distances.csv")
        # copy sum-user distances
        if os.path.exists(f"{auxilary_distance_path}/{model}/sum_user_distances.csv"):
            shutil.copyfile(f"{auxilary_distance_path}/{model}/sum_user_distances.csv",
                            f"{distance_path}/{model}/sum_user_distances.csv")
        else:
            print(f"file {auxilary_distance_path}/{model}/sum_user_distances.csv not found")


if __name__ == '__main__':
    app()
