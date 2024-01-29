import os

import pandas as pd
from scipy.stats import entropy
import torch
from torchmetrics.text import InfoLM

from tqdm.auto import tqdm

from egises import Egises, PersevalParams

from collections import Counter

# distance measures
from rouge_score import rouge_scorer
from nltk.translate import meteor
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load
import numpy as np
import warnings
import typer
import utils

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_SET_PATH = f"{CURRENT_DIR}/dataset"
PERSONALIZED_MODELS = ("NAML_1", "NRMS_1", "NRMS_2", "EBNR_1", "EBNR_2")
NON_PERSONALIZED_MODELS_LIST = ("big_bird", "brio", "prophetnet", "cls", "t5_base")

warnings.filterwarnings('ignore')

app = typer.Typer()

CONSOLIDATED_FILEPATH = f"{CURRENT_DIR}/dataset/final_tokenized_consolidated_data.jsonl"
SCORES_PATH = f"{CURRENT_DIR}/scores"

# load infoLM model only once
# TODO: load model based on function argument
device = 'cuda' if torch.cuda.is_available() else 'cpu'
infolm = InfoLM('google/bert_uncased_L-2_H-128_A-2', idf=False, device=device, alpha=1.0, beta=1.0,
                information_measure="ab_divergence",
                verbose=False)

bertscore = load("bertscore")


def calculate_meteor(texts):
    text1, text2 = texts
    tokens1 = text1.split()
    tokens2 = text2.split()
    result = meteor([tokens1], tokens2)
    return round(result, 5)


def calculate_bleu(texts):
    text1, text2 = texts
    tokens1 = text1.split(" ")
    tokens2 = text2.split(" ")
    result = sentence_bleu([tokens1], tokens2)
    # print(round(result,5))
    return round(result, 5)


def calculate_rougeL(texts):
    text1, text2 = texts
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    result = scorer.score(text1, text2)["rougeL"].fmeasure
    # print(round(result,5))
    return round(result, 5)


def calculate_rougeSU4(texts):
    candidate, reference = texts
    candidate = candidate.split(" ")
    reference = reference.split(" ")
    # Calculate skip-bigram matches upto 5 gram
    for i in range(5):
        candidate_ngrams = [tuple(candidate[j:j + i + 1]) for j in range(len(candidate) - i)]
        reference_ngrams = [tuple(reference[j:j + i + 1]) for j in range(len(reference) - i)]
    # Calculate the number of skip-n-gram matches
    match_count = sum((Counter(candidate_ngrams) & Counter(reference_ngrams)).values())
    # Calculate the number of skip-ngrams in the candidate and reference summaries
    candidate_bigram_count = len(candidate_ngrams)
    reference_bigram_count = len(reference_ngrams)
    # Calculate precision, recall, and F-measure
    precision = match_count / candidate_bigram_count if candidate_bigram_count > 0 else 0.0
    recall = match_count / reference_bigram_count if reference_bigram_count > 0 else 0.0
    beta = 1  # Set beta to 1 for ROUGE-SU4
    f_measure = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall) if (
                                                                                                   precision + recall) > 0 else 0.0
    # return precision, recall, f_measure
    return round(f_measure, 5)


def _text2distribution(text: list, common_vocab: set):
    """
    Calculate the probability distribution of words in the given text with respect to the common vocabulary.

    Parameters:
    - text: List of words.
    - common_vocab: Common vocabulary list.

    Returns:
    - prob_dist: Probability distribution represented as a numpy array.
    """
    word_counts = Counter(text)
    total_words = len(text)

    # Initialize probability distribution with zeros
    prob_dist = np.zeros(len(common_vocab))
    if total_words == 0:
        return prob_dist
    # Populate the probability distribution based on the common vocabulary
    for i, word in enumerate(common_vocab):
        prob_dist[i] = word_counts[word] / total_words

    return prob_dist


def calculate_JSD(texts):
    # JSD calculation without OOVs
    # create common vocab
    tokens_1, tokens_2 = [text.split() for text in texts]
    common_vocab = set(tokens_1).union(set(tokens_2))

    # calculate probability distributions
    p_dist = _text2distribution(tokens_1, common_vocab)
    q_dist = _text2distribution(tokens_2, common_vocab)

    m_dist = 0.5 * (p_dist + q_dist)

    # Calculate Kullback-Leibler divergences
    kl_p = entropy(p_dist, m_dist, base=2)
    kl_q = entropy(q_dist, m_dist, base=2)

    # Calculate Jensen-Shannon Divergence
    jsd_value = 0.5 * (kl_p + kl_q)
    jsd_value = round(jsd_value, 4)
    return jsd_value


def calculate_infoLM(texts: list):
    pred, target = texts
    score = infolm([pred], [target]).item()
    return round(score, 5)


def calculate_bert_score(texts: list):
    pred, target = texts
    score = bertscore.compute(predictions=[pred], references=[target], lang="en", model_type="distilbert-base-uncased",
                              device='cuda')
    return score['f1'][0]


def calculate_hj(texts: list):
    # calculated from survey submissions from db. Refer export_hj_data_to_csv function in data_preprocessing.py
    raise Exception("Not implemented")


@app.command()
def populate_distances(model_name: str, distance_measure: str, max_workers: int = 1):
    """
    model_name: one of PERSONALIZED_MODELS or NON_PERSONALIZED_MODELS_LIST
    distance_measure: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd
    max_workers: number of workers to use for multiprocessing
    """
    measure_dict = {
        "meteor": calculate_meteor,
        "bleu": calculate_bleu,
        "rougeL": calculate_rougeL,
        "rougeSU4": calculate_rougeSU4,
        "infoLM": calculate_infoLM,
        "JSD": calculate_JSD,
        "bert_score": calculate_bert_score,
        "hj": calculate_hj
    }

    if distance_measure == "infoLM" and max_workers > 1:
        print(f"setting max_workers to 1 for infoLM")
        max_workers = 1

    try:
        assert distance_measure in measure_dict.keys()
        measure = measure_dict[distance_measure]
    except AssertionError as err:
        print(f"measure should be one of {measure_dict.keys()}")
        return
    eg = Egises(model_name=model_name, measure=measure,
                documents=utils.get_model_documents(model_name, CONSOLIDATED_FILEPATH),
                score_directory=f"{SCORES_PATH}/{measure.__name__}/{model_name}",
                max_workers=max_workers)
    eg.populate_distances()


@app.command()
def generate_scores(distance_measure: str, sampling_freq: int = 10, max_workers: int = 1, simplified_flag: bool = False,
                    stability: bool = False, version="v2"):
    """
    distance_measure: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd
    sampling_freq: sampling frequency for percentage less than 100
    max_workers: number of workers to use for multiprocessing
    version: generate suffixed scores files to avoid overwriting
    saves scores in scores/distance_measure/egises_scores_version.csv
    """
    measure_dict = {
        "meteor": calculate_meteor,
        "bleu": calculate_bleu,
        "rougeL": calculate_rougeL,
        "rougeSU4": calculate_rougeSU4,
        "infoLM": calculate_infoLM,
        "JSD": calculate_JSD,
        "bert_score": calculate_bert_score,
        "hj": calculate_hj
    }

    if distance_measure == "infoLM" and max_workers > 1:
        print(f"setting max_workers to 1 for infoLM")
        max_workers = 1

    try:
        assert distance_measure in measure_dict.keys()
        measure = measure_dict[distance_measure]
    except AssertionError as err:
        print(f"measure should be one of {measure_dict.keys()}")
        return

    egises_csv_path = f"{SCORES_PATH}/{measure.__name__}/egises_scores_{version}.csv"
    accuracy_csv_path = f"{SCORES_PATH}/{measure.__name__}/accuracy_scores_{version}.csv"

    # measure = calculate_meteor
    for model_name in tqdm([*PERSONALIZED_MODELS, *NON_PERSONALIZED_MODELS_LIST]):
        distance_directory = f"{SCORES_PATH}/{measure.__name__}/{model_name}"
        # for model_name in tqdm([*PERSONALIZED_MODELS]):
        model_egises_tuple, model_accuracy_tuple = [model_name], [model_name]
        eg = Egises(model_name=model_name, measure=measure,
                    documents=utils.get_model_documents(model_name, CONSOLIDATED_FILEPATH),
                    score_directory=distance_directory, max_workers=max_workers, version=version)
        eg.populate_distances(simplified_flag=simplified_flag)

        if stability:
            header_range = range(100, 10, -20)
            header = ["models", *list(range(100, 10, -20)), "bias", "variance"]
        else:
            header_range = range(100, 110, 10)
            header = ["models", "100", "bias", "variance"]

        for sample_percentage in header_range:
            print(f"calculating for {model_name} with sample percentage {sample_percentage}")
            if sample_percentage == 100:
                eg_score, accuracy_score = eg.get_egises_score(sample_percentage=sample_percentage)
                print(f"eg_score: {eg_score}, accuracy_score: {accuracy_score}")
            else:
                # for sample percentage less than 100, calculate score 10 times and take mean
                eg_scores = []
                accuracy_scores = []
                pbar = tqdm(range(sampling_freq))
                for i in range(sampling_freq):
                    eg_score, accuracy_score = eg.get_egises_score(sample_percentage=sample_percentage)
                    eg_scores.append(eg_score)
                    accuracy_scores.append(accuracy_score)
                    pbar.update(1)
                pbar.close()
                eg_score = round(np.mean(eg_scores), 4)
                accuracy_score = round(np.mean(accuracy_scores), 4)
                print(f"eg_score: {eg_score}, accuracy_score: {accuracy_score}")
            model_egises_tuple.append(eg_score)
            model_accuracy_tuple.append(accuracy_score)

        std = np.std(model_egises_tuple[1:])
        # calculate vaBaseExceptionriance of model_tuple[1:]
        var = np.var(model_egises_tuple[1:])
        model_egises_tuple.append(round(std, 4))
        model_egises_tuple.append(var)

        std = np.std(model_accuracy_tuple[1:])
        # calculate vaBaseExceptionriance of model_tuple[1:]
        var = np.var(model_accuracy_tuple[1:])
        model_accuracy_tuple.append(round(std, 4))
        model_accuracy_tuple.append(var)

        print(f"model_egises_tuple: {model_egises_tuple}")
        print(f"model_accuracy_tuple: {model_accuracy_tuple}")
        utils.write_scores_to_csv([model_egises_tuple],
                                  fields=header,
                                  filename=egises_csv_path)

        utils.write_scores_to_csv([model_accuracy_tuple],
                                  fields=header,
                                  filename=accuracy_csv_path)
    accuracy_df = pd.read_csv(accuracy_csv_path)
    egises_df = pd.read_csv(egises_csv_path)
    return accuracy_df, egises_df


@app.command()
def generate_perseval_scores(distance_measure: str, sampling_freq: int = 10, max_workers: int = 1,
                             simplified_flag: bool = False, stability: bool = False, EDP_beta: float = 1.0,
                             version="v2"):
    """
    distance_measure: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd
    model_name: sampling frequency for percentage less than 100
    max_workers: number of workers to use for multiprocessing
    simplified_flag: if True, calculate proportions without using doc based normalization
    stability: if True, calculate by sampling different percentage records
    version: generate suffixed scores files to avoid overwriting
    saves scores in scores/distance_measure/perseval_scores_version.csv
    """
    measure_dict = {
        "meteor": calculate_meteor,
        "bleu": calculate_bleu,
        "rougeL": calculate_rougeL,
        "rougeSU4": calculate_rougeSU4,
        "infoLM": calculate_infoLM,
        "JSD": calculate_JSD,
        "bert_score": calculate_bert_score,
        "hj": calculate_hj,

    }

    if distance_measure == "infoLM" and max_workers > 1:
        print(f"setting max_workers to 1 for infoLM")
        max_workers = 1

    try:
        assert distance_measure in measure_dict.keys()
        measure = measure_dict[distance_measure]
    except AssertionError as err:
        print(f"measure should be one of {measure_dict.keys()}")
        return
    # measure = calculate_meteor
    accuracy_csv_path = f"{SCORES_PATH}/{measure.__name__}/perseval_accuracy_scores_{version}_simp_{simplified_flag}.csv"
    perseval_csv_path = f"{SCORES_PATH}/{measure.__name__}/perseval_scores_{version}_simp_{simplified_flag}.csv"
    for model_name in tqdm([*PERSONALIZED_MODELS, *NON_PERSONALIZED_MODELS_LIST]):
        distance_directory = f"{SCORES_PATH}/{measure.__name__}/{model_name}"
        # for model_name in tqdm([*PERSONALIZED_MODELS]):
        model_perseval_tuple, model_accuracy_tuple = [model_name], [model_name]
        eg = Egises(model_name=model_name, measure=measure,
                    documents=utils.get_model_documents(model_name, CONSOLIDATED_FILEPATH),
                    score_directory=distance_directory, max_workers=max_workers, version=version)
        eg.populate_distances(simplified_flag=simplified_flag)

        perseval_params = PersevalParams(EDP_beta=EDP_beta)
        print(f"calculating for {model_name} with perseval params {perseval_params}")

        if stability:
            header_range = range(100, 10, -20)
            header = ["models", *list(range(100, 10, -20)), "bias", "variance"]
        else:
            header_range = range(100, 110, 10)
            header = ["models", "100", "bias", "variance"]

        for sample_percentage in header_range:
            print(f"sample percentage:{sample_percentage}")
            if sample_percentage == 100:
                perseval_score, accuracy_score = eg.get_perseval_score(sample_percentage=sample_percentage,
                                                                       perseval_params=perseval_params)
                print(f"perseval_score@{sample_percentage}%: {perseval_score}, accuracy_score: {accuracy_score}")
            else:
                # for sample percentage less than 100, calculate score 10 times and take mean
                perseval_scores = []
                accuracy_scores = []
                pbar = tqdm(range(sampling_freq))
                for i in range(sampling_freq):
                    perseval_score, accuracy_score = eg.get_perseval_score(sample_percentage=sample_percentage,
                                                                           perseval_params=perseval_params)
                    perseval_scores.append(perseval_score)
                    accuracy_scores.append(accuracy_score)
                    pbar.update(1)
                pbar.close()
                perseval_score = round(np.mean(perseval_scores), 4)
                accuracy_score = round(np.mean(accuracy_scores), 4)
                print(f"perseval_score@{sample_percentage}%: {perseval_score}, accuracy_score: {accuracy_score}")
            model_perseval_tuple.append(perseval_score)
            model_accuracy_tuple.append(accuracy_score)

        std = np.std(model_perseval_tuple[1:])
        # calculate vaBaseExceptionriance of model_tuple[1:]
        var = np.var(model_perseval_tuple[1:])
        model_perseval_tuple.append(round(std, 4))
        model_perseval_tuple.append(var)

        std = np.std(model_accuracy_tuple[1:])
        # calculate vaBaseExceptionriance of model_tuple[1:]
        var = np.var(model_accuracy_tuple[1:])
        model_accuracy_tuple.append(round(std, 4))
        model_accuracy_tuple.append(var)

        print(f"model_perseval_tuple: {model_perseval_tuple}")
        print(f"model_accuracy_tuple: {model_accuracy_tuple}")

        utils.write_scores_to_csv([model_perseval_tuple],
                                  fields=header,
                                  filename=perseval_csv_path)

        utils.write_scores_to_csv([model_accuracy_tuple],
                                  fields=header,
                                  filename=accuracy_csv_path)
    accuracy_df = pd.read_csv(accuracy_csv_path)
    perseval_df = pd.read_csv(perseval_csv_path)
    return accuracy_df, perseval_df


@app.command()
def calculate_correlation(dmeasure_1: str, dmeasure_2: str, pmeasure1: str = "perseval", pmeasure2: str = "perseval",
                          m1_version="final",
                          m2_version="final"):
    """
    dmeasure_1: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd
    dmeasure_2: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd, hj
    pmeasure1: one of egises, perseval, degress, perseval_accuracy
    pmeasure2: one of egises, perseval, degress, perseval_accuracy
    """
    assert pmeasure1 in ["egises", "perseval", "perseval_accuracy", "degress"]
    assert pmeasure2 in ["egises", "perseval", "perseval_accuracy", "degress"]
    assert dmeasure_1 in ["meteor", "bleu", "rougeL", "rougeSU4", "infoLM", "JSD", "hj", "bert_score"]
    assert dmeasure_2 in ["meteor", "bleu", "rougeL", "rougeSU4", "infoLM", "JSD", "hj", "bert_score"]

    measure1_dict = utils.get_measure_scores(measure=dmeasure_1, p_measure=pmeasure1, version=m1_version)
    measure2_dict = utils.get_measure_scores(measure=dmeasure_2, p_measure=pmeasure2, version=m2_version)
    corr_dict = utils.get_correlation_from_model_dict(measure1_dict, measure2_dict)
    return corr_dict


@app.command()
def get_borda_scores(dmeasure_1: str = "infoLM", dmeasure_2: str = "rougeL", p1_measure: str = "perseval",
                     p2_measure: str = "perseval_accuracy", m1_version="v2",
                     m2_version="v2") -> dict:
    """
    dmeasure_1: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd
    dmeasure_2: one of meteor, bleu, rougeL, rougeSU4, infoLM, jsd, hj
    p_measure: one of egises, perseval_accuracy, perseval, degress
    """
    assert p1_measure in ["egises", "perseval", "perseval_accuracy"]
    assert p2_measure in ["egises", "perseval", "perseval_accuracy"]
    assert dmeasure_1 in ["meteor", "bleu", "rougeL", "rougeSU4", "infoLM", "JSD", "hj"]
    assert dmeasure_2 in ["meteor", "bleu", "rougeL", "rougeSU4", "infoLM", "JSD", "hj"]

    dmeasure_1_dict = utils.get_measure_scores(measure=dmeasure_1, p_measure=p1_measure, version=m1_version)
    dmeasure_2_dict = utils.get_measure_scores(measure=dmeasure_2, p_measure=p2_measure, version=m2_version)

    sorted_dmeasure_1_dict = dict(sorted(dmeasure_1_dict.items(), key=lambda item: - item[1]))
    sorted_dmeasure_2_dict = dict(sorted(dmeasure_2_dict.items(), key=lambda item: - item[1]))

    rank1 = list(sorted_dmeasure_1_dict.keys())
    rank2 = list(sorted_dmeasure_2_dict.keys())
    borda_dict = utils.calculate_borda_consensus(rank1, rank2)
    # print(f"borda_dict: {borda_dict}")
    return borda_dict


if __name__ == "__main__":
    app()
    # for acc_measure in ["bleu", "meteor", "rougeL", "rougeSU4", "infoLM"]:
    #     bk = get_borda_scores(dmeasure_1="hj", dmeasure_2=acc_measure, p1_measure="perseval",
    #                           p2_measure="perseval_accuracy", m1_version="v2", m2_version="v2")
    #     # print(f"bk:{bk}")
    #     bk = dict(sorted(bk.items(), key=lambda item: item[1]))
    #     bk = {key: i for i, key in enumerate(bk.keys(), 1)}
    #     # print(f"bk:{bk}")
    #     hj_scores = _get_measure_scores(measure="hj", p_measure="perseval", version="v2")
    #     hj_scores = dict(sorted(hj_scores.items(), key=lambda item: item[1]))
    #     # print(f"hj_scores:{hj_scores}")
    #     hj_scores = {key: i for i, key in enumerate(hj_scores.keys(), 1)}
    #     # print(f"hj_scores:{hj_scores}")
    #     corr_dict = _get_correlation_from_model_dict(bk, hj_scores)
    #     print(f"corr_dict_{acc_measure}:{corr_dict}")
    # hj_scores =
    # bk = _calculate_borda_consensus(["a", "c", "b"], ["a", "b", "c"])
    # print(bk)

    # calculate correlation between 2 measures
    # measure_dict = {
    #     "rougeL": calculate_rougeL,
    #     "rougeSU4": calculate_rougeSU4,
    #     "meteor": calculate_meteor,
    #     "bleu": calculate_bleu,
    #     "infoLM": calculate_infoLM,
    #     "JSD": calculate_JSD,
    #     "bert_score": calculate_bert_score,
    # }
    # measure_list = ["rougeL", "rougeSU4", "meteor", "bleu", "infoLM", "bert_score", "JSD"]
    # measure_pearson_list, spearman_list, kendal_list = [], [], []
    # for measure in measure_dict.keys():
    #     print(f"measure: {measure}")
    #     corr_dict = calculate_correlation(dmeasure_1=measure, dmeasure_2=measure, pmeasure1="degress",
    #                                       pmeasure2="perseval", m1_version="sfinal",
    #                                       m2_version="final")
    #     for corr_method in corr_dict.keys():
    #         # print(f"{measure}_hj_{corr_method}:{corr_dict[corr_method]}")
    #         print(f"{corr_dict[corr_method]}")
    #     print(f"*" * 50)
    # print(f"{measure}_perseval_{measure}-hj_perseval_rouge:{corr_dict}")

    # for ablation studies
    # for edp_beta in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
    #     generate_perseval_scores(distance_measure="infoLM", sampling_freq=10, max_workers=1, stability=False,
    #                              EDP_beta=edp_beta, version=f"ablation_{edp_beta}")
    #     generate_perseval_scores(distance_measure="hj", sampling_freq=10, max_workers=1, stability=False,
    #                              EDP_beta=edp_beta, version=f"ablation_{edp_beta}")

    # for edp_beta in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
    #     corr_dict = calculate_correlation(dmeasure_1="infoLM", dmeasure_2="hj", pmeasure1="perseval",
    #                                           pmeasure2="perseval", m1_version=f"ablation_{edp_beta}_simp_False",
    #                                           m2_version=f"ablation_{edp_beta}_simp_False")
    #     for corr_method in corr_dict.keys():
    #         # print(f"{measure}_hj_{corr_method}:{corr_dict[corr_method]}")
    #         print(f"{corr_dict[corr_method]}")
    #     print(f"*" * 50)
    # corr_dict = calculate_correlation(dmeasure_1="JSD", dmeasure_2="hj", pmeasure1="degress",
    #                                   pmeasure2="perseval", m1_version=f"final",
    #                                   m2_version=f"final")
    # for corr_method in corr_dict.keys():
    #     # print(f"{measure}_hj_{corr_method}:{corr_dict[corr_method]}")
    #     print(f"{corr_dict[corr_method]}")
    # print(f"*" * 50)
