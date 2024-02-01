import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser, Namespace

from results_parser import ResultsParser

# model,pretrained_or_tuned,multilingual_or_french,number_params,size_gb,seq_len,embedding_dim,model_type,license
CHARACTERISTICS = {
    "pretrained_or_tuned": "categorical",
    "multilingual_or_french": "categorical",
    "number_params": "numerical",
    "size_gb": "numerical",
    "seq_len": "numerical",
    "embedding_dim": "numerical",
    "model_type": "categorical",
    "license": "categorical",
}


def parse_args() -> Namespace:
    """Parse command line arguments

    Returns:
        (argparse.Namespace): the arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--results_folder", required=True, type=str)
    parser.add_argument("--characteristics_csv", required=True, type=str)
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./script_mteb_french/results_analysis/performance_vs_characteristics_plots",
    )
    args = parser.parse_args()

    return args


def perfomance_vs_characteristic_plot(
    results_df: pd.DataFrame,
    characteristics_df: pd.DataFrame,
    target_characteristic: str,
    characteristic_type: str,
    output_path: str,
    mode: str = "avg",
):
    data = results_df.assign(**results_df.iloc[:, 1:].rank(pct=True))
    data = data.melt(id_vars="model", var_name="dataset", value_name="score")
    data = data[["model", "score"]]
    if mode == "avg":
        data = data.groupby("model").mean().reset_index()
    data = data.merge(characteristics_df, on="model", how="left")
    data = data.dropna(axis=0, how="any")
    # Set seaborn style
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    plt.title(f"Performance vs {target_characteristic}")
    plt.xlabel(target_characteristic)
    plt.ylabel("Score")
    if characteristic_type == "categorical":
        sns.scatterplot(data=data, x=target_characteristic, y="score")
    elif characteristic_type == "numerical":
        sns.scatterplot(data=data, x=target_characteristic, y="score")
        plt.xscale("log")
    else:
        raise ValueError(f"Unknown characteristic type: {characteristic_type}")
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    args = parse_args()
    rp = ResultsParser()
    results_df = rp(args.results_folder, return_main_scores=False)
    results_df = results_df.drop(
        columns=[
            ("BitextMining", "DiaBLaBitextMining"),
            ("BitextMining", "FloresBitextMining"),
        ]
    )
    results_df = results_df.droplevel(0, axis=1)
    results_df = results_df.reset_index()
    results_df["model"] = results_df["model"].apply(
        lambda x: x.replace(args.results_folder, "")
    )
    # this should not be necessary with final csv
    results_df = results_df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    characteristics_df = pd.read_csv(args.characteristics_csv)
    results_df.merge(characteristics_df, on="model", how="left")
    for k, v in CHARACTERISTICS.items():
        output_path = os.path.join(args.output_folder, f"perf_vs_{k}_avg.png")
        perfomance_vs_characteristic_plot(
            results_df, characteristics_df, k, v, output_path, mode="avg"
        )
        output_path = os.path.join(args.output_folder, f"perf_vs_{k}_all.png")
        perfomance_vs_characteristic_plot(
            results_df, characteristics_df, k, v, output_path, mode="all"
        )
