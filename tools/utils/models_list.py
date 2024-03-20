SENTENCE_TRANSORMER_MODELS = [
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "flaubert/flaubert_base_uncased",
    "flaubert/flaubert_base_cased",
    "flaubert/flaubert_large_cased",
    "dangvantuan/sentence-camembert-base",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/LaBSE",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-small",
    "distilbert-base-uncased",
    "Geotrend/distilbert-base-25lang-cased",
    "Geotrend/distilbert-base-en-fr-es-pt-it-cased",
    "Geotrend/distilbert-base-en-fr-cased",
    "Geotrend/distilbert-base-fr-cased",
    "Geotrend/bert-base-25lang-cased",
    "Geotrend/bert-base-15lang-cased",
    "Geotrend/bert-base-10lang-cased",
    "shibing624/text2vec-base-multilingual",
    "izhx/udever-bloom-560m",
    "izhx/udever-bloom-1b1",
    "sentence-transformers/sentence-t5-base",
    "sentence-transformers/sentence-t5-large",
    "sentence-transformers/sentence-t5-xl",
    "sentence-transformers/sentence-t5-xxl",
    "intfloat/e5-mistral-7b-instruct",
    "Wissam42/sentence-croissant-llm-base"
]

# these models max_length is indicated to be 514 whereas the embedding layer actually supports 512
SENTENCE_TRANSORMER_MODELS_WITH_ERRORS = [
    "camembert/camembert-base",
    "camembert/camembert-large",
    "dangvantuan/sentence-camembert-large",
    "xlm-roberta-base",
    "xlm-roberta-large",
]

UNIVERSAL_SENTENCE_ENCODER_MODELS = [
    "vprelovac/universal-sentence-encoder-multilingual-3",
    "vprelovac/universal-sentence-encoder-multilingual-large-3",
]
# TODO: use json file keys

LASER_MODELS = ["laser2"]

VOYAGE_MODELS = ["voyage-2", "voyage-code-2"]

OPEN_AI_MODELS = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]

COHERE_MODELS = ["embed-multilingual-light-v3.0", "embed-multilingual-v3.0"]

MISTRAL_MODELS = ["mistral-embed"]

TYPES_TO_MODELS = {
    "sentence_transformer": SENTENCE_TRANSORMER_MODELS
    + SENTENCE_TRANSORMER_MODELS_WITH_ERRORS,
    "universal_sentence_encoder": UNIVERSAL_SENTENCE_ENCODER_MODELS,
    "laser": LASER_MODELS,
    "voyage_ai": VOYAGE_MODELS,
    "open_ai": OPEN_AI_MODELS,
    "cohere": COHERE_MODELS,
    "mistral_ai": MISTRAL_MODELS,
}
