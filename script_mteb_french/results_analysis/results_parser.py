import os
import warnings
import json
from argparse import ArgumentParser, Namespace
from mteb import MTEB
from mteb.abstasks import AbsTask
import pandas as pd

DATASET_KEYS = {
    "DiaBlaBitextMining": ["fr-en"],
    "FloresBitextMining": MTEB(tasks=['FloresBitextMining'], task_langs=['fr', 'en']).tasks[0].langs,
    "MasakhaNEWSClassification": MTEB(tasks=['MasakhaNEWSClassification'], task_langs=['fr']).tasks[0].langs,
    "MasakhaNEWSClusteringS2S": MTEB(tasks=['MasakhaNEWSClusteringS2S'], task_langs=['fr']).tasks[0].langs,
    "MasakhaNEWSClusteringP2P": MTEB(tasks=['MasakhaNEWSClusteringP2P'], task_langs=['fr']).tasks[0].langs,
    "XPQARetrieval": MTEB(tasks=['XPQARetrieval'], task_langs=['fr']).tasks[0].langs,
}

HF_SUBSETS_VALUES = ["fra-fra"]
ISO3_LANGUAGE = ["fra-Latn"]

MODELS_TO_IGNORE = ['voyage-01', 'voyage-02', 'voyage-lite-01', 'Geotrend/distilbert-base-en-fr-es-pt-it-cased', 
                    'Geotrend/bert-base-10lang-cased', 'Geotrend/bert-base-15lang-cased', 'Geotrend/bert-base-25lang-cased',
                    'dangvantuan/sentence-camembert-large', 'distilbert-base-uncased']


class ResultsParser:
    """A class to parse the results of MTEB evaluations
    """
    def __init__(self, lang:str="fr") -> None:
        self.lang = lang

        self.tasks_main_scores_map = ResultsParser.get_tasks_attribute("main_score")
        self.tasks_main_scores_map["SyntecRetrieval"] = "ndcg_at_10" # Quick fix to match old results. Sould modify main metric for syntecretrieval in mteb module
        self.tasks_type_map = ResultsParser.get_tasks_attribute("type")
        self.eval_splits_map = ResultsParser.get_tasks_attribute("eval_splits")


    def __call__(self, results_folder:str, apply_style:bool=False, save_results:bool=True, return_main_scores:bool=False, **kwargs) -> pd.DataFrame:
        """Wrapper function that crawls through the results folder and build
        a dataframe reprensenting all the results found.

        Args:
            results_folder (str): path to the result folder

        kwargs:
            format (str): format to save the results to. One of excel, csv or latex. Defaults to "excel".

        Returns:
            pd.DataFrame: the results as a dataframe
        """
        result_dict = ResultsParser._load_json_files(results_folder)
        results_df, tasks_main_scores_subset = self._convert_to_results_dataframe(result_dict)
        results_df = self._add_multiindex_to_df(results_df)
        
        if "process_raw" in kwargs and kwargs["process_raw"]:
            processed_results = ResultsParser.process_raw(results_df, **kwargs)
            if apply_style:
                processed_results = ResultsParser._add_style_to_df(processed_results)
            if save_results:
                ResultsParser._save_as_file(processed_results, raw=False, **kwargs)
        
        if apply_style:
            results_df = ResultsParser._add_style_to_df(results_df)
        if save_results:
            ResultsParser._save_as_file(results_df, raw=True, **kwargs)
        if return_main_scores:
            return results_df, tasks_main_scores_subset

        return results_df

        
    @staticmethod
    def _load_json_files(root_folder:str) -> dict:
        """Loads all the json files located in the folder
        (typically containing results) and concatenate them
        in a dict.

        Args:
            root_folder (str): _description_

        Returns:
            dict: the results as a dict
        """
        result_dict = {}

        for root, _, files in os.walk(root_folder):
            for file in files:
                if file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    dir_path = os.path.dirname(full_path).replace(root_folder + "/", '')
                    file_name_without_extension = os.path.splitext(file)[0]
                    with open(full_path, 'r') as json_file:
                        json_content = json.load(json_file)

                    if dir_path not in result_dict:
                        result_dict[dir_path] = {}
                    result_dict[dir_path][file_name_without_extension] = json_content

        return result_dict


    @staticmethod
    def get_tasks_attribute(attribute:str='main_score') -> dict[str:str]:
        """Runs through the MTEB module and
        gets the attribute value of each task.

        Args:
            attribute (str): the name of the attribute. Must belong to the
                "metadata" property of the task.

        Returns:
            dict: a mapping with keys being the tasks names 
                and values being the attributes. Defaults to 'main_score'
        """
        tasks_categories_cls = [cls for cls in AbsTask.__subclasses__()]
        tasks_cls = [
            cls(langs=["fr"])
            for cat_cls in tasks_categories_cls
            for cls in cat_cls.__subclasses__()
            if cat_cls.__name__.startswith("AbsTask")
        ]
        tasks_dict = {cls.metadata.name: cls for cls in tasks_cls}
        tasks_attribute = {k: vars(v.metadata)[attribute] for k, v in tasks_dict.items()}

        return tasks_attribute


    def _get_task_score(self, task_name:str, task_results:str, subkey:str|None = None, split: str|None = None) -> tuple[float, tuple[str, str]]:
        """Considering a task, gets its results

        Args:
            task_name (str): the name of the task. e.g. "SickFr"
            task_results (str): the results of that task. e.g the content of the json result file
            split (str, optional): the split to get the results from. Defaults to 'test'.

        Returns:
            result (float): the value of the metric for obtained for that task
            result_name_score (tuple[str, str]): the name of the task and name of the main scoring metric 
                for that task
        """
        selected_split = split if split else self.split

        if task_results["mteb_version"].startswith("1.11.1"):
            result = None
            for eval in task_results["scores"][selected_split]:
                hf_subset = eval['hf_subset']
                languages = eval['languages'] # used when hf_subset = "default"
                if (hf_subset == subkey) or (hf_subset in HF_SUBSETS_VALUES) or (languages == ISO3_LANGUAGE):
                    result = eval["main_score"]
                    continue
            main_score = self.tasks_main_scores_map[task_name]
            result_name_score = (task_name, main_score)
            return result, result_name_score
        
        key = subkey if subkey else self.lang
        result = task_results[selected_split]
        if key in result:
            result = result[key]
        if task_name in self.tasks_main_scores_map:
            main_score = self.tasks_main_scores_map[task_name]
            if main_score in result:
                result = result[main_score]
                result_name_score = (task_name, main_score)
            elif main_score == "cosine_spearman":
                result = result['cos_sim']['spearman']
                result_name_score = (task_name, "cosine_spearman")
            elif main_score == "ap":
                result = result['cos_sim']['ap']
                result_name_score = (task_name, "cosine_ap")
            else:
                result = None
                result_name_score = (task_name, None)
        else:
            warnings.warn(f"Task name '{task_name}' not found in MTEB module.")

        return result, result_name_score


    def _convert_to_results_dataframe(self, result_dict:dict):
        """Converts the results from a dict to a dataframe

        Args:
            result_dict (dict): the result dict returned from load_json_files()

        Returns:
            pd.DataFrame: the results as a df
        """
        results_records = []
        tasks_main_scores_subset = []
        for model_name, model_results in result_dict.items():
            model_ignore = any([x in model_name for x in MODELS_TO_IGNORE])
            if not model_ignore:
                for task_name, task_results in model_results.items():
                    if task_name in self.tasks_type_map:
                        task_type = self.tasks_type_map[task_name]
                        if task_name in DATASET_KEYS:
                            subkeys = DATASET_KEYS[task_name]
                        else:
                            subkeys = [None]
                        for split in self.eval_splits_map[task_name]:
                            if (split in task_results) or ("scores" in task_results and split in task_results["scores"]):
                                for subkey in subkeys:
                                    result, result_name_score = self._get_task_score(task_name, task_results, subkey, split)
                                    dataset_name = f"{task_name}_{split}_{subkey}" if subkey and task_type == "BitextMining" else f"{task_name}_{split}"
                                    self.tasks_type_map[dataset_name] = task_type
                                    results_records.append({'model': model_name, 'dataset': dataset_name, 'result': result})
                                    tasks_main_scores_subset.append(result_name_score)
                    else:
                        warnings.warn(f"Task name '{task_name}' not found in MTEB module.")
        results_df = pd.DataFrame.from_records(results_records)
        results_pivot = results_df.pivot(index='model', columns='dataset', values='result')
        tasks_main_scores_subset = dict(tasks_main_scores_subset)

        return results_pivot, tasks_main_scores_subset
    

    def _add_multiindex_to_df(self, results_df:pd.DataFrame) -> pd.DataFrame:
        """Adds a multiindex to columns of the results df
        with the task type

        Args:
            results_df (pd.DataFrame): df returned from _convert_to_results_dataframe()

        Returns:
            pd.DataFrame: the df with a secondary column index
        """
        # reorder column by task type
        reordered_col_names = [col for col in self.tasks_type_map if col in results_df.columns]
        results_df = results_df[reordered_col_names]
        # add multiindex
        multiindex_col_names = [(v,k) for k,v in self.tasks_type_map.items() if k in results_df.columns]
        results_df.columns = pd.MultiIndex.from_tuples(multiindex_col_names)

        return results_df
    
    @staticmethod
    def _add_style_to_df(results_df:pd.DataFrame) -> pd.io.formats:
        """Adds style to the results df.
        - centers values
        - bold the max value of each column

        Args:
            result_df (pd.DataFrame): the result df from add_multiindex_to_df()

        Returns:
            pd.io.formats.style.Styler: the styled df
        """
        def highlight_max(s):
            '''
            highlight the maximum in a Series yellow.
            '''
            is_max = s == s.max()
            return ['font-weight: bold' if v else '' for v in is_max]

        style = results_df.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        style.set_table_styles({"model": [{"selector": "th", "props": [("text-align", "left")]}]})
        style.set_properties(**{'text-align': 'center'})
        style.apply(highlight_max)

        return style
    
    @staticmethod
    def _save_as_file(results_df: pd.DataFrame, output_format: str = "excel", output_folder: str = "./", raw: bool = True, **kwargs):
        if output_format not in ["excel", "latex", "csv"]:
            raise ValueError(f"'format' argument should be either excel, latex or csv, not {format}")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        filename = "results_raw" if raw else "results"
        
        match output_format:
            case "csv":
                results_df.style.clear()
                results_df.to_csv(os.path.join(output_folder, f"{filename}.csv"))
            case "excel":
                results_df.to_excel(os.path.join(output_folder, f"{filename}.xlsx"))
            case "latex":
                results_df.to_latex(os.path.join(output_folder, f"{filename}.tex"))
        print(f"Saved results in {output_format} format at {os.path.join(output_folder, filename)}.")
    
    @staticmethod
    def process_raw(results_df: pd.DataFrame, **kwargs) -> pd.DataFrame: 
        results_df.style.clear()
        results = results_df.copy()
        # Remove columns if multiple test split for task
        results.drop([('PairClassification', 'OpusparcusPC_validation.full'),], axis=1, inplace=True)
        # Remove bitext mining -> use in separate table
        cols2remove = [
            (task_type, task_name) for task_type, task_name in results.columns
            if "XPQA" in task_name or "Mintaka" in task_name
            or task_type == "BitextMining"
            ]
        results.drop(cols2remove, axis=1, inplace=True)
        results.columns = pd.MultiIndex.from_tuples([(task_type, task_name.replace("_test", "").replace("_test.full", "")) for task_type, task_name in results.columns])
        results.sort_index(axis=1, level=[0, 1], ascending=[True, False], inplace=True)
        results = results.round(2)
        results = results.fillna("")

        return results


def parse_args() -> Namespace:
    """Parse command line arguments

    Returns:
        (argparse.Namespace): the arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--results_folder", required=True, type=str)
    parser.add_argument("--output_format", type=str, choices=["excel", "csv", "latex"], default="excel")
    parser.add_argument("--apply_style", type=bool, default=True)
    parser.add_argument("--output_folder", type=str, default="./analyses_outputs/")
    parser.add_argument("--process_raw", type=bool, default=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    rp = ResultsParser()
    results_df = rp(**dict(args._get_kwargs()))