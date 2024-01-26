import json
import matplotlib.pyplot as plt
import seaborn as sns

from .results_parser import ResultsParser

if __name__ == '__main__':
    # Get results
    results_folder_path = '../results'
    rp = ResultsParser()
    results_df, tasks_main_scores_subset = rp(results_folder_path, return_main_scores=True, format="csv")
    # Dataset correlations
    spearman_corr_matrix_datasets = results_df.corr(method='spearman')
    spearman_corr_matrix_datasets.to_csv('correlation_analysis/spearman_corr_matrix_datasets.csv')
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr_matrix_datasets, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Dataset Correlation Heatmap (Spearman)')
    plt.savefig('correlation_analysis/spearman_corr_heatmap_datasets.png', bbox_inches='tight')
    with open('correlation_analysis/main_scores.json', 'w') as f:
        json.dump(tasks_main_scores_subset, f, indent=4)
    # Model correlations
    transposed_results_df = results_df.transpose()
    spearman_corr_matrix_models = transposed_results_df.corr(method='spearman')
    spearman_corr_matrix_models.to_csv('correlation_analysis/spearman_corr_matrix_models.csv')
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr_matrix_models, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Model Correlation Heatmap (Spearman)')
    plt.savefig('correlation_analysis/spearman_corr_heatmap_models.png', bbox_inches='tight')
