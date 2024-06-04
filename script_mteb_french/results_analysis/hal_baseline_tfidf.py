from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from datasets import load_dataset
from sklearn.metrics import classification_report

DATASET = "lyon-nlp/clustering-hal-s2s"
SEED = 42

STOPWORDS = stopwords.words("french") + stopwords.words("english")

dataset = load_dataset(DATASET, name="mteb_eval", split="test")
dataset = dataset.class_encode_column("domain")
id2label = {k: dataset.features["domain"].int2str(k) for k in range(dataset.features["domain"].num_classes)}
dataset = dataset.train_test_split(test_size=0.3, shuffle=True, stratify_by_column="domain", seed=SEED)

X_train, y_train = dataset["train"]["title"], dataset["train"]["domain"]
X_test, y_test = dataset["test"]["title"], dataset["test"]["domain"]

estimators = [LogisticRegression(), RandomForestClassifier(), SVC()]
parameters = [
    {
        "estimator__C": [0.1, 1, 10],
    },
    {
        "estimator__n_estimators": [100, 200, 300],
        "estimator__max_features": ["auto", "sqrt", "log2"],
        "estimator__bootstrap": [True, False],
        "estimator__class_weight": ["balanced", "balanced_subsample"],
    },
    {
        "estimator__C": [0.1, 1, 10],
        "estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
    }
]


for estimator, params in zip(estimators, parameters):
    print(f"Estimator: {estimator}")   
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words=STOPWORDS, ngram_range=(1, 3))),
        ("estimator", estimator)
    ])
     
    grid = GridSearchCV(pipeline, param_grid={ 
        **params,
        "tfidf__max_features": [1000, 5000, 10000],
        "tfidf__use_idf": [True, False],
        }, 
        cv=3, n_jobs=10, verbose=2
    )

    grid.fit(X_train, y_train)
    score = grid.score(X_test, y_test)
    print(f"Accuracy: {score}")
    report = classification_report(y_test, grid.predict(X_test), labels=range(len(id2label)), target_names=id2label.values())
    print(report)