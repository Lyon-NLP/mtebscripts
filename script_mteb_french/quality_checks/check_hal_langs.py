import langdetect
from datasets import load_dataset

DATASET = "lyon-nlp/clustering-hal-s2s"

dataset = load_dataset(DATASET, name="mteb_eval", split="test")

langs = [langdetect.detect(sample.lower()) for sample in dataset["title"]]

dataset = dataset.add_column(name="lang", column=langs)

df = dataset.to_pandas()
print(df.lang.value_counts())

df.to_csv("hal_s2s_lang.csv", index=False)