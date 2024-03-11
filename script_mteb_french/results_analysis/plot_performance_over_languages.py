import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("performance_analysis/models_ranks_en_fr.csv")

    x = df.model.values
    x_axis = np.arange(len(x))
    y_1 = df.avg_perf_fr.values
    y_2 = df.avg_perf_en.values

    cmap = plt.get_cmap("Set2")
    plt.bar(x_axis - 0.2, y_1, 0.4, label='French', color=cmap(0))
    plt.bar(x_axis + 0.2, y_2, 0.4, label='English', color=cmap(1))

    plt.xticks(x_axis, x, rotation=80)
    plt.ylim([30, 75])
    plt.ylabel("Overall avergae performance")
    plt.tight_layout()
    plt.legend(loc='lower left')
    plt.savefig(f"performance_analysis/avg_performance_overall.png")


if __name__ =="__main__":
    main()