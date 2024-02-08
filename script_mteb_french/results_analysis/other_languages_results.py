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
    fr_results, tasks_main_scores = res_parser(args.results_folder, return_main_scores=True)
    print(tasks_main_scores)

    task_names = [task_name for _, task_name in set(fr_results.columns.values)]
    models_name_to_index = {split_model_name(idx): idx for idx in fr_results.index}

    df_mc = pd.read_csv("models_characteristics/models_characteristics.csv")
    
    mteb_res = {}

    for model_name, is_multilingual in zip(df_mc['model'].values, df_mc['multilingual_or_french']):
        model_name = split_model_name(model_name)

        if is_multilingual == "multilingual":
            try:
                mteb_res[model_name] = load_dataset("mteb/results", model_name)
                logging.info(f"Downloaded model {model_name} results")

                mteb_results = mteb_res[model_name]['test']
                print(len(mteb_results))
                for idx in range(len(mteb_results)):
                    try:
                        tasks_avg_multi_score = {}
                        for task_name in task_names:
                            task_multi_score = []
                            if task_name == mteb_results[idx]["mteb_dataset_name"]:
                                if mteb_results[idx]['metric'] == tasks_main_scores[task_name]:
                                    pass
                                task_multi_score.append()

                                print(mteb_results[idx]["mteb_dataset_name"])
                                print(mteb_results[idx]['eval_language'])
                                print(mteb_results[idx]['metric'])
                                print(mteb_results[idx]['score'])
                            
                            tasks_avg_multi_score[task_name] = sum(task_multi_score) / len(task_multi_score)
                    except ValueError as e:
                        logging.error(e)
            except ValueError:
                logging.error(f"Model {model_name} not evaluated on other languages")
        


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_folder", type=str, default='results')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)