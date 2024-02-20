"""
Script for merging results of mteb-fr with mteb-original 

Need to clone mteb/results repo first :
git lfs install
git clone https://huggingface.co/datasets/mteb/results

Then run:
cd mtebscripts
cp -r ~/results/results ./mteb_results

mteb_reults folder is already ignored by .gitignore
"""


import os
import sys
import logging
import argparse
import json
import shutil

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


SPLITS = [
    'test',
    'validation', 
    'val',
    'dev',
    'devtest'
]

def split_model_name(model_name: str):
    model_name = model_name.split('/')
    if len(model_name) == 2:
        model_name = model_name[1]
    else:
        model_name = model_name[0]
    return model_name


def get_task_result_for_fr(map_: dict[str, dict[str, dict]], model_name: str, task_name: str) -> dict:
    return map_[model_name][task_name]
    

def load_tasks(root_folder: str, return_path_map: bool = False):
    result_dict = {}
    
    if return_path_map:
        paths_map = {}
    
    for root, _, files in os.walk(root_folder):
        for file_ in files:
            if file_.endswith(".json"):
                full_path = os.path.join(root, file_)
                dir_path = os.path.dirname(full_path).replace(root_folder + "/", '')
                #file_name_without_extension = os.path.splitext(file_)[0]
                with open(full_path, 'r') as json_file:
                    json_content = json.load(json_file)

                dir_path = split_model_name(dir_path) # keep only model name without prefix
                if dir_path not in result_dict:
                    result_dict[dir_path] = {}
                result_dict[dir_path][file_] = json_content
                if return_path_map:
                    if dir_path not in paths_map:
                        paths_map[dir_path] = {}
                    paths_map[dir_path][file_] = full_path
                    paths_map[dir_path]['path'] = full_path.replace("/" + file_, '')
    
    if return_path_map:
        return result_dict, paths_map
    
    return result_dict


def add_fr_result_to_file(root_folder: str, model_name: str, task_name: str, split: str, results: dict):
    res_file = os.path.join(root_folder, model_name, task_name)
    
    logging.info("Load results file and add fr results")
    with open(res_file) as fp:
        orig_file = json.load(fp)
    
    orig_file[model_name][task_name][split]['fr'] = results
    
    logging.info("Write new result file")
    with open(res_file) as fp:
        json.dump(orig_file, fp)


def create_new_model_evaluation_folder(root_folder: str, model_name: str):
    path = os.path.join(root_folder, model_name)
    if not os.path.exists(path):
        os.mkdir(path)


def create_new_task_file(root_folder_orig,  model_name, task_name: str = None, mteb_fr_paths_map: dict = None, copy_all_tasks: bool = False):
    if copy_all_tasks:
        model_fr_results = mteb_fr_paths_map[model_name]['path']
        model_origin = os.path.join(root_folder_orig, model_name)
        # copy all fr results folder for model to mteb_orignal model folder
        try:
            shutil.copytree(src=model_fr_results, dst=model_origin) 
        except FileExistsError:
            shutil.rmtree(model_origin)   
            shutil.copytree(src=model_fr_results, dst=model_origin)
    else:
        fr_res_file_path = mteb_fr_paths_map[model_name][task_name]
        dst_mteb_orig = os.path.join(root_folder_orig, model_name)
        # copy fr results task file to mteb_orignal model folder
        shutil.copy(src=fr_res_file_path, dst=dst_mteb_orig)
    

def main(args): 
    #TODO handle lowercase and model names: laser2 == LASER2
    base_path_mteb_orig = args.mteb_results_folder
    base_path_mteb_fr = args.mteb_fr_results_folder
    
    mteb_fr_map, mteb_fr_paths_map = load_tasks(root_folder=base_path_mteb_fr, return_path_map=True)
    mteb_orig_map = load_tasks(root_folder=base_path_mteb_orig)
    
    for model, results in mteb_fr_map.items():
        if model in mteb_orig_map.keys(): # if model already evaluated by mteb original
            for task in results.keys(): # for every evaluated task 
                if task in mteb_orig_map[model].keys(): # if task exists in mteb original
                    results_orig = get_task_result_for_fr(map_=mteb_orig_map, model_name=model, task_name=task)
                    results_fr = get_task_result_for_fr(map_=mteb_fr_map, model_name=model, task_name=task)
                    
                    for split in results_fr.keys(): # get results of all splits
                        try: # check if task has split in original mteb
                            if 'fr' not in results_orig[split].keys(): # do not modify existing results
                                logging.info(f"Add fr results to existing eval of {model} for task {task}")
                                add_fr_result_to_file(base_path_mteb_orig, model, task, split, results_fr)
                        except:
                            logging.warning(f"Task {task} has no attribute language fr")
                else:
                    logging.info(f"Copy {task} file from mteb-fr results to mteb-original model {model} folder")
                    create_new_task_file(mteb_fr_paths_map=mteb_fr_paths_map, root_folder_orig=base_path_mteb_orig, model_name=model, task_name=task)
        else:
            logging.info(f"The model {model} is only evaluated in mteb-fr. Copy all folder to mteb-original")
            create_new_model_evaluation_folder(root_folder=base_path_mteb_orig, model_name=model)
            create_new_task_file(mteb_fr_paths_map=mteb_fr_paths_map, root_folder_orig=base_path_mteb_orig, model_name=model, copy_all_tasks=True)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mteb_results_folder", type=str, default="mteb_results")
    parser.add_argument("--mteb_fr_results_folder", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)