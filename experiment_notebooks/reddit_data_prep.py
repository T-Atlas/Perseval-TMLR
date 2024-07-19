from collections import namedtuple

import fire
import pandas as pd
from datasets import load_dataset

HfPartition = namedtuple('hf_partition', 'dataset subset split')


def get_master_data_from_hf(hf_dataset="openai/summarize_from_feedback", subset="axis", split="validation"):
    """
    gets all news data from a given hf_dataset
    :param hf_dataset:
    :param subset:
    :param split:
    :return:
    """
    if subset:
        ds = load_dataset(hf_dataset, subset)
    else:
        ds = load_dataset(hf_dataset)

    df = ds[split].to_pandas()
    print(df.columns)
    unique_users = df["worker"].drop_duplicates().shape
    print(f"unique_users: {unique_users}")
    unique_docs = df["info"].drop_duplicates().shape
    print(f"unique_docs: {unique_docs}")
    # flatten data
    #     Category: category
    # Headline: actual title
    # News Body: document text
    df["News ID"] = df["info"].apply(lambda x: x["id"])
    df["News Body"] = df["info"].apply(lambda x: x["post"])
    df["Headline"] = df["info"].apply(lambda x: x["title"])
    df["Category"] = df["info"].apply(lambda x: x["subreddit"])
    df = df[~df["Category"].isna()]
    df = df[["News ID", "News Body", "Headline", "Category"]]
    df = df.drop_duplicates("News ID")
    return df


def get_master_data_for_inference():
    partitions = [
        HfPartition(dataset="openai/summarize_from_feedback", subset="axis", split="validation"),
        HfPartition(dataset="openai/summarize_from_feedback", subset="comparisons", split="validation"),
    ]
    dfs = []
    for p in partitions:
        df = get_master_data_from_hf(hf_dataset=p.dataset, subset=p.subset, split=p.split)
        dfs.append(df)
    master_df = pd.concat(dfs, axis=0)  # df_axis.append(df_comparison, ignore_index=True)
    master_df = master_df.drop_duplicates("News ID")
    master_df.to_csv("inference_news.tsv", sep="\t", index=False)


def get_master_data_for_training(outdir="train_news.tsv"):
    """
    for training, we use comparison|train (click history) and axis|test (user summaries)
    :param hf_dataset:
    :param outdir:
    :return:
    """
    partitions = [
        HfPartition(dataset="openai/summarize_from_feedback", subset="axis", split="test"),
        HfPartition(dataset="openai/summarize_from_feedback", subset="comparisons", split="train"),
    ]
    dfs = []
    for p in partitions:
        df = get_master_data_from_hf(hf_dataset=p.dataset, subset=p.subset, split=p.split)
        dfs.append(df)
    master_df = pd.concat(dfs, axis=0)  # df_axis.append(df_comparison, ignore_index=True)
    master_df = master_df.drop_duplicates("News ID")
    master_df.to_csv(outdir, sep="\t", index=False)


### user summaries and clicks related

def get_user_summaries(hf_dataset="openai/summarize_from_feedback", subset="axis", split="validation",
                       rating_summ_threshold=5.0):
    if subset:
        ds_history = load_dataset(hf_dataset, subset)
    else:
        ds_history = load_dataset(hf_dataset)

    df = ds_history[split].to_pandas()
    # flatten
    df = df.rename(columns={"worker": "uid"})
    print(df.columns)
    df["doc_id"] = df["info"].apply(lambda x: x["id"])

    df["summary_text"] = df["summary"].apply(lambda x: x["text"])
    df["summary_rating"] = df["summary"].apply(lambda x: x["axes"]["overall"])
    df["summary_model"] = df["summary"].apply(lambda x: x["policy"])
    df = df[["uid", "doc_id", "summary_model", "summary_text", "summary_rating"]]
    idx = df.groupby(['doc_id', 'uid'])['summary_rating'].idxmax()
    df = df.loc[idx]

    # filter above threshold as clicks
    df_usummaries = df[df["summary_rating"] > rating_summ_threshold]
    df_usummaries = df_usummaries[["uid", "doc_id", "summary_text"]]
    # print(df_usummaries.columns)
    return df_usummaries


def get_user_clicks(hf_dataset="openai/summarize_from_feedback", subset="comparisons", split="validation",
                    rating_click_threshold=5.0):
    if subset:
        ds_history = load_dataset(hf_dataset, subset)
    else:
        ds_history = load_dataset(hf_dataset)
    df_history = ds_history[split].to_pandas()
    # flatten
    df_history = df_history.rename(columns={"worker": "uid"})
    df_history["doc_id"] = df_history["info"].apply(lambda x: x["id"])
    df_history["doc_text"] = df_history["info"].apply(lambda x: x["post"])
    df_history["subreddit"] = df_history["info"].apply(lambda x: x["subreddit"])

    # filter out non subreddit
    df_history = df_history[~df_history["subreddit"].isna()]

    df_history["confidence"] = df_history["extra"].apply(lambda x: x["confidence"])
    history_candidates = df_history[["uid", "doc_id", "confidence"]].groupby(["doc_id", "uid"]).aggregate("mean")[
        "confidence"]

    history_candidates = history_candidates.reset_index()
    history_candidates = history_candidates[history_candidates["confidence"] > rating_click_threshold]
    history_candidates = history_candidates[["uid", "doc_id"]]

    return history_candidates


def get_user_clicksum_for_training(hf_dataset="openai/summarize_from_feedback", click_subset="comparisons",
                                   click_split="validation", summ_subset="axis", summ_split="test",
                                   outdir="personalized_train.tsv"):
    """
    We use comparisons|train (click history) and axis|test (user summaries) for training
    :param hf_dataset:
    :param click_subset:
    :param click_split:
    :param summ_subset:
    :param summ_split:
    :return:
    """
    user_summaries = get_user_summaries(hf_dataset=hf_dataset, subset=summ_subset, split=summ_split)
    print(f"user_summaries.columns: {user_summaries.columns}")
    user_clicks = get_user_clicks(hf_dataset=hf_dataset, subset=click_subset, split=click_split)
    print(f"user_clicks.columns: {user_clicks.columns}")

    # exclude user summary docs from click docs
    user_doc_tuples = [tuple(x) for x in user_summaries[["uid", "doc_id"]].values]
    user_clicks["summary_flag"] = user_clicks.apply(lambda x: (x["uid"], x["doc_id"]) in user_doc_tuples, axis=1)
    user_clicks = user_clicks[~user_clicks["summary_flag"]]
    user_clicks = user_clicks.groupby("uid").aggregate({"doc_id": lambda x: ",".join(list(x.to_list()))})
    user_clicks = user_clicks.reset_index()
    user_clicks = user_clicks.rename(columns={"doc_id": "clicknewsID"})
    # user_clicks

    # print(f"user_summaries.columns: {user_summaries.columns}")
    user_summaries = user_summaries.groupby("uid").aggregate(
        {"doc_id": lambda x: ",".join(x.to_list()), "summary_text": lambda x: ";;".join(x.to_list())}, axis=1)
    user_summaries = user_summaries.reset_index()
    user_summaries = user_summaries.rename(columns={"doc_id": "posnewID", "summary_text": "rewrite_titles"})

    personalized_df = pd.merge(user_clicks, user_summaries, on=["uid"])
    personalized_df = personalized_df.rename(columns={"uid": "userid"})
    personalized_df.to_csv(outdir, sep="\t", index=False)
    print(personalized_df.columns)
    print(personalized_df.columns)
    print(f"saved to {outdir}")
    return personalized_df


if __name__ == '__main__':
    fire.Fire()
