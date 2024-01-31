import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import os


def run_statistical_tests(data: pd.DataFrame, output_path: str):
    results_lists = list(data.values[:, 1:])
    friedman_stats = friedmanchisquare(*results_lists)
    if friedman_stats.pvalue < 0.05:
        print(
            f"There is a significant difference between the models (p-value: {friedman_stats.pvalue}). Running post-hoc tests..."
        )
        data_melted = data.melt(id_vars="model", var_name="dataset", value_name="score")
        avg_rank = (
            data_melted.groupby("dataset")
            .score.rank(pct=True)
            .groupby(data_melted.model)
            .mean()
        )
        detailed_test_results = sp.posthoc_conover_friedman(
            data_melted,
            melted=True,
            block_col="dataset",
            group_col="model",
            y_col="score",
        )
        plt.figure(figsize=(10, 8))
        plt.title("Post hoc conover friedman tests")
        sp.sign_plot(detailed_test_results)
        plt.savefig(
            os.path.join(output_path, "conover_friedman.png"), bbox_inches="tight"
        )
        plt.figure(figsize=(10, 6))
        plt.title("Critical difference diagram of average score ranks")
        sp.critical_difference_diagram(avg_rank, detailed_test_results)
        plt.savefig(
            os.path.join(output_path, "critical_difference_diagram.png"),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    # TODO: use latest csv instead
    data = pd.read_csv("correlation_analysis/results_table.csv")
    # this should not be necessary with final csv
    data = data.fillna(0)
    run_statistical_tests(data, "results_analysis/")
