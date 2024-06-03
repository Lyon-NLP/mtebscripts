from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from datasets import load_dataset
from sklearn.metrics import classification_report

DATASET = "lyon-nlp/clustering-hal-s2s"
SEED = 42

dataset = load_dataset(DATASET, name="mteb_eval", split="test")
dataset = dataset.remove_columns(["hal_id"])

dataset = dataset.filter(lambda example: example["domain"] not in ("image"))
dataset = dataset.class_encode_column("domain")

dataset = dataset.train_test_split(test_size=0.3, shuffle=True, stratify_by_column="domain", seed=SEED)

X_train, y_train = dataset["train"]["title"], dataset["train"]["domain"]
X_test, y_test = dataset["test"]["title"], dataset["test"]["domain"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("logistic", LogisticRegression(max_iter=1000, random_state=SEED))
])

pipeline.fit(X_train, y_train)

score = pipeline.score(X_test, y_test)
print(f"Accuracy: {score}")

report = classification_report(y_test, pipeline.predict(X_test))
print(report)