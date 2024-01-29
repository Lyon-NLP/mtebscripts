import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from results_analysis import ResultsParser

RESULTS_DIR = "results/"
PLOTS_DIR = "plots"
TASK_TYPES = [
    "Classification",
    "Clustering",
    "PairClassification",
    "Retrieval",
    "Reranking",
    "STS",
    "Summarization"
]

def plot_df_per_task_type(df: pd.DataFrame, fig_name: str):
    avg_column = "average_score"
    df[avg_column] = df.mean(axis=1)
    df.dropna(subset=[avg_column], inplace=True)
    df.sort_values(by=avg_column, ascending=True, inplace=True)
    model_names = df.index.tolist()
    x = [x.replace("results/", "") for x in model_names]
    height = df[avg_column]

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.barh(x, height)
    ax.bar_label(bars)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.title(fig_name.replace("png", ""))
    plt.savefig(f"{os.path.join(PLOTS_DIR, fig_name)}")


def main(args):
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)

    rp = ResultsParser()
    results = rp(RESULTS_DIR)

    task_type = args.task_type
    if task_type and task_type in TASK_TYPES:
        fig_name = f"{task_type}.png"
        filtered_results = results.loc[:][task_type]
        plot_df_per_task_type(df=filtered_results, fig_name=fig_name)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type", type=str, default="Classification", choices=TASK_TYPES
    )
    parser.add_argument("--model_name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
