import os
import sys
import argparse
import logging

from datasets import load_dataset
import pandas as pd

from results_parser import ResultsParser

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def split_model_name(model_name: str):
    model_name = model_name.split('/')
    if len(model_name) == 2:
        model_name = model_name[1]
    else:
        model_name = model_name[0]
    return model_name


def main(args):
    res_parser = ResultsParser()
    fr_results = res_parser(args.results_folder)
    
    task_types = [task_type for task_type, _ in set(fr_results.columns.values)]

    models_name_to_index = {idx: split_model_name(idx) for idx in fr_results.index}

    fr_results_avg = fr_results.copy()
    new_df = pd.DataFrame({})

    for task_type in set(task_types):
        filtered_results = fr_results_avg.loc[:][task_type]
        new_df[f"avg_{task_type}"] = filtered_results.mean(axis=1)
    
    overall_avg = new_df.mean(axis=1)
    new_df.insert(0, "overall_avg", overall_avg)

    new_df.reset_index(inplace=True)
    models_short_name = new_df.model.apply(lambda x: models_name_to_index[x])
    new_df.insert(1, "model_short", models_short_name)

    # Sort models from best to worse based on overall average performance score
    new_df.sort_values(by=['overall_avg'], ascending=False, inplace=True)
    new_df.reset_index(drop=True, inplace=True)

    # Get rank of models
    new_df.reset_index(inplace=True)
    new_df.rename(columns={'index': 'rank'}, inplace=True)
    new_df['rank'] = new_df['rank'].apply(lambda x: x+1)

    # Save results to CSV
    output_dir = "performance_analysis"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    new_df.to_csv(os.path.join(output_dir, "mteb_fr_avg_perfromance.csv"), index=False)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_folder", type=str, default='results')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)