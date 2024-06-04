from nltk.corpus import stopwords
import os
import argparse
import json

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import nltk
nltk.download('stopwords')

DATASET = "lyon-nlp/clustering-hal-s2s"
STOPWORDS = stopwords.words("french") + stopwords.words("english")


def main(args):
    dataset = load_dataset(DATASET, name="mteb_eval", split="test")
    dataset = dataset.class_encode_column("domain")
    id2label = {k: dataset.features["domain"].int2str(
        k) for k in range(dataset.features["domain"].num_classes)}
    dataset = dataset.train_test_split(
        test_size=0.3, shuffle=True, stratify_by_column="domain", seed=args.dataset_seed)

    X_train, y_train = dataset["train"]["title"], dataset["train"]["domain"]
    X_test, y_test = dataset["test"]["title"], dataset["test"]["domain"]

    estimators = [LogisticRegression(max_iter=1000), RandomForestClassifier(), SVC()]
    parameters = [
        {
            "estimator__C": [0.1, 1, 10],
        },
        {
            "estimator__n_estimators": [200, 300],
            "estimator__bootstrap": [True, False],
            "estimator__class_weight": ["balanced", "balanced_subsample"],
        },
        {
            "estimator__C": [0.1, 1, 10],
            "estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "estimator__class_weight": [True, False],
        }
    ]

    for estimator, params in zip(estimators, parameters):
        print(f"Estimator: {estimator}")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True,
             stop_words=STOPWORDS, ngram_range=(1, 3))),
            ("estimator", estimator)
        ])

        grid = GridSearchCV(pipeline, param_grid={
            **params,
            "tfidf__max_features": [1000, 5000, 10000]
        },
            cv=3, n_jobs=10, verbose=2
        )

        grid.fit(X_train, y_train)
        score = grid.score(X_test, y_test)
        print(f"Accuracy: {score}")
        report = classification_report(
            y_test, grid.predict(X_test),
            labels=range(len(id2label)),
            target_names=id2label.values(),
            output_dict=True
        )

        if not os.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        with open(os.path.join(args.output_dir, f"report_{estimator.__class__.__name__}_{args.dataset_seed}.json"), "w") as f:
            json.dump(report, f, indent=4)

        with open(os.path.join(args.output_dir, f"params_{estimator.__class__.__name__}_{args.dataset_seed}.json"), "w") as f:
            json.dump(grid.best_params_, f, indent=4)

        print(grid.best_estimator_)
        print(grid.best_params_)
        print(grid.best_score_)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="tfidf_baseline")
    parser.add_argument("--dataset_seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
