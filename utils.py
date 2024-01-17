import csv
import os
import pickle
import random
import re
import traceback

import nltk
import pandas as pd
import typer
import ujson
from tqdm.auto import tqdm
import utils
from egises import Summary, Document

DATA_SET_PATH = "dataset"
SCORES_PATH = f"scores"


def write_scores_to_csv(rows, fields=None, filename="scores.csv"):
    # print(type(rows))
    if fields:
        try:
            assert rows and len(rows[0]) == len(fields)
        except AssertionError as err:
            print(traceback.format_exc())
            print(f"fields: {fields}")
            print(rows[0])

            return
    if os.path.exists(filename):
        # append to existing file
        with open(filename, 'a') as f:
            # using csv.writer method from CSV package
            if fields:
                write = csv.writer(f)
            write.writerows(rows)
    else:
        with open(filename, 'w') as f:
            # using csv.writer method from CSV package
            if fields:
                write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)


def load_data(path):
    with open(path + '.pkl', 'rb') as file:
        var = pickle.load(file)
    return var


def _tokenize(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # wordnet lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()

    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation

    text = re.sub(r'[\d+]', '', text.lower())  # remove numerical values and convert to lower case

    tokens = nltk.word_tokenize(text)  # tokenization

    tokens = [token for token in tokens if token not in stopwords]  # removing stopwords

    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # lemmatization

    # my_string= " ".join(tokens)

    return tokens


def _tokenize_text(text):
    return " ".join(_tokenize(text))


def _get_measure_df(measure: str = "", p_measure: str = "", version: str = "v2"):
    version = f"_{version}" if version else ""
    csv_file = f"{SCORES_PATH}/calculate_{measure}/{p_measure}_scores{version}.csv"
    df = pd.read_csv(csv_file)
    return df


def get_measure_scores(measure: str = "", p_measure: str = "", version: str = "v2"):
    if p_measure == "degress":
        p_measure = "egises"
        df = _get_measure_df(measure, p_measure, version)
        df = df[["models", "100"]]
        df = df.set_index(["models"])
        measure_dict = df.to_dict(orient="index")
        measure_dict = {item[0]: 1 - item[1]["100"] for item in measure_dict.items()}
        return measure_dict

    df = _get_measure_df(measure, p_measure, version)
    df = df[["models", "100"]]
    df = df.set_index(["models"])
    measure_dict = df.to_dict(orient="index")
    measure_dict = {item[0]: item[1]["100"] for item in measure_dict.items()}
    return measure_dict


def get_model_documents(model_name, filepath=f"{DATA_SET_PATH}/consolidated_data.jsonl", measure=""):
    with open(filepath, "r") as fpr:
        for line in fpr.readlines():
            line = line.strip()
            line = ujson.loads(line)
            doc_id, doc_text, doc_summ = line["doc_id"], line["doc_text"], line["doc_summ"]
            user_summaries = [Summary("user", doc_id, uid, model_summary_map["user"]) for uid, model_summary_map in
                              line["m_summary_dict"].items()]
            model_summaries = [Summary(model_name, doc_id, uid, model_summary_map[model_name]) for
                               uid, model_summary_map in line["m_summary_dict"].items()]
            yield Document(doc_id, doc_text, doc_summ, user_summaries, model_summaries)


# correlation related code
def get_correlation_from_model_dict(model1: dict, model2: dict):
    sorted_measure1_dict = dict(sorted(model1.items(), key=lambda item: item[1]))

    # print(f"sorted_measure1_dict: {sorted_measure1_dict}")
    measure1_list = list(sorted_measure1_dict.values())
    measure2_list = [model2[model] for model in sorted_measure1_dict.keys()]
    measure1_list = pd.Series(measure1_list)
    measure2_list = pd.Series(measure2_list)

    # Calculating correlation
    corr_types = ['pearson', 'kendall', 'spearman']
    corr_dict = {corr_type: round(measure1_list.corr(measure2_list, method=corr_type), 5) for corr_type in corr_types}
    return corr_dict


def calculate_borda_consensus(rank1: list, rank2: list) -> dict:
    """
    rank1: list of models in order of rank
    rank2: list of models in order of rank
    """
    assert len(rank1) == len(rank2)
    n = len(rank1)
    rank1_dict = {model: n - i for i, model in enumerate(rank1)}
    rank2_dict = {model: n - i for i, model in enumerate(rank2)}
    borda_dict = {}
    for model in rank1_dict.keys():
        borda_dict[model] = rank1_dict[model] + rank2_dict[model]
    borda_dict = dict(sorted(borda_dict.items(), key=lambda item: item[1]))
    return borda_dict
