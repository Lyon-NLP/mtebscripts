from datasets import load_dataset

import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

DATASET = "lyon-nlp/clustering-hal-s2s"
SEED = 42
STOPWORDS = stopwords.words("french") # + stopwords.words("english") 


dataset = load_dataset(DATASET, name="mteb_eval", split="test")
dataset = dataset.class_encode_column("domain")
num_classes = dataset.features["domain"].num_classes
id2label = {k: dataset.features["domain"].int2str(k) for k in range(num_classes)}
dataset = dataset.train_test_split(test_size=0.3, shuffle=True, stratify_by_column="domain", seed=SEED)

X_train, y_train = dataset["train"]["title"], dataset["train"]["domain"]
X_test, y_test = dataset["test"]["title"], dataset["test"]["domain"]

nlp = spacy.load('fr_core_news_sm', disable = ['parser','ner'])

docs_train = nlp.pipe(X_train)
docs_test = nlp.pipe(X_test)

def tokenize_text(doc):
    return [token.lemma_.lower() for token in doc if token not in STOPWORDS]

tokenized_X_train = [tokenize_text(doc) for doc in docs_train]
tokenized_X_test = [tokenize_text(doc) for doc in docs_test]
print(tokenized_X_train[:5])

common_dictionary = Dictionary(tokenized_X_train)
common_corpus = [common_dictionary.doc2bow(text) for text in tokenized_X_train]

lda = LdaModel(
    common_corpus, 
    num_topics=num_classes, 
    id2word=common_dictionary, 
    eval_every=5, random_state=SEED)

print(f"Perplexity: {lda.log_perplexity(common_corpus)}")

coherence_lda = CoherenceModel(
    model=lda, texts=tokenized_X_train,
    dictionary=common_dictionary, coherence='c_v')
print(f"Coherence: {coherence_lda.get_coherence()}")