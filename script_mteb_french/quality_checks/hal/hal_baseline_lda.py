from datasets import load_dataset

import spacy
import nltk
import spacy.cli
nltk.download('stopwords')
from nltk.corpus import stopwords

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

DATASET = "lyon-nlp/clustering-hal-s2s"
SEED = 42
STOPWORDS = stopwords.words("french") # + stopwords.words("english") 

try:
    nlp = spacy.load('fr_core_news_sm', disable = ['parser','ner'])
except FileNotFoundError:
    spacy.cli.download("fr_core_news_sm")
    nlp = spacy.load('fr_core_news_md', disable = ['parser','ner'])


dataset = load_dataset(DATASET, name="mteb_eval", split="test")
dataset = dataset.class_encode_column("domain")
num_classes = dataset.features["domain"].num_classes
id2label = {k: dataset.features["domain"].int2str(k) for k in range(num_classes)}

texts, domains = dataset["title"], dataset["domain"]

docs = nlp.pipe(texts)

def tokenize_text(doc):
    return [token.lemma_.lower() for token in doc if token not in STOPWORDS]

tokenized_texts = [tokenize_text(doc) for doc in docs]
print(tokenized_texts[:5])

common_dictionary = Dictionary(tokenized_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in tokenized_texts]

lda = LdaModel(
    common_corpus, 
    num_topics=num_classes, 
    id2word=common_dictionary, 
    eval_every=5, random_state=SEED)

print(f"Perplexity: {lda.log_perplexity(common_corpus)}")

coherence_lda = CoherenceModel(
    model=lda, texts=tokenized_texts,
    dictionary=common_dictionary, coherence='c_v')
print(f"Coherence: {coherence_lda.get_coherence()}")