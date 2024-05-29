from datasets import load_dataset
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import json
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


dataset_fr = load_dataset("lyon-nlp/summarization-summeval-fr-p2p", "test")
dataset_en = load_dataset("mteb/summeval", "test")


bleu_scorer = BLEU(effective_order=True)
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# bleu scores
all_bleus_vs_human_fr = []
all_bleus_vs_text_fr = []
for i in range(len(dataset_fr["test"]["machine_summaries"])):
    for j in range(len(dataset_fr["test"]["machine_summaries"][i])):
        bleu_score = bleu_scorer.sentence_score(
            hypothesis=dataset_fr["test"]["machine_summaries"][i][j],
            references=dataset_fr["test"]["human_summaries"][i],
        )
        all_bleus_vs_human_fr.append(bleu_score.score)
        bleu_score = bleu_scorer.sentence_score(
            hypothesis=dataset_fr["test"]["machine_summaries"][i][j],
            references=[dataset_fr["test"]["text"][i]],
        )
        all_bleus_vs_text_fr.append(bleu_score.score)

all_bleus_vs_human_en = []
all_bleus_vs_text_en = []
for i in range(len(dataset_en["test"]["machine_summaries"])):
    for j in range(len(dataset_en["test"]["machine_summaries"][i])):
        bleu_score = bleu_scorer.sentence_score(
            hypothesis=dataset_en["test"]["machine_summaries"][i][j],
            references=dataset_en["test"]["human_summaries"][i],
        )
        all_bleus_vs_human_en.append(bleu_score.score)
        bleu_score = bleu_scorer.sentence_score(
            hypothesis=dataset_en["test"]["machine_summaries"][i][j],
            references=[dataset_en["test"]["text"][i]],
        )
        all_bleus_vs_text_en.append(bleu_score.score)

# rouge scores

rouge1_vs_text_fr = []
rouge2_vs_text_fr = []
rougeL_vs_text_fr = []
rouge1_vs_human_fr = []
rouge2_vs_human_fr = []
rougeL_vs_human_fr = []
for i in range(len(dataset_fr["test"]["machine_summaries"])):
    for j in range(len(dataset_fr["test"]["machine_summaries"][i])):
        rouge_scores = rouge.score(
            dataset_fr["test"]["machine_summaries"][i][j], dataset_fr["test"]["text"][i]
        )
        rouge1_vs_text_fr.append(rouge_scores["rouge1"].recall)
        rouge2_vs_text_fr.append(rouge_scores["rouge2"].recall)
        rougeL_vs_text_fr.append(rouge_scores["rougeL"].recall)
        for k in range(len(dataset_fr["test"]["human_summaries"][i])):
            rouge_scores = rouge.score(
                dataset_fr["test"]["machine_summaries"][i][j],
                dataset_fr["test"]["human_summaries"][i][k],
            )
            rouge1_vs_human_fr.append(rouge_scores["rouge1"].recall)
            rouge2_vs_human_fr.append(rouge_scores["rouge2"].recall)
            rougeL_vs_human_fr.append(rouge_scores["rougeL"].recall)

rouge1_vs_text_en = []
rouge2_vs_text_en = []
rougeL_vs_text_en = []
rouge1_vs_human_en = []
rouge2_vs_human_en = []
rougeL_vs_human_en = []
for i in range(len(dataset_en["test"]["machine_summaries"])):
    for j in range(len(dataset_en["test"]["machine_summaries"][i])):
        rouge_scores = rouge.score(
            dataset_en["test"]["machine_summaries"][i][j], dataset_en["test"]["text"][i]
        )
        rouge1_vs_text_en.append(rouge_scores["rouge1"].recall)
        rouge2_vs_text_en.append(rouge_scores["rouge2"].recall)
        rougeL_vs_text_en.append(rouge_scores["rougeL"].recall)
        for k in range(len(dataset_en["test"]["human_summaries"][i])):
            rouge_scores = rouge.score(
                dataset_en["test"]["machine_summaries"][i][j],
                dataset_en["test"]["human_summaries"][i][k],
            )
            rouge1_vs_human_en.append(rouge_scores["rouge1"].recall)
            rouge2_vs_human_en.append(rouge_scores["rouge2"].recall)
            rougeL_vs_human_en.append(rouge_scores["rougeL"].recall)

# saving data

data = {
    "bleu": {
        "fr": {"vs_human": all_bleus_vs_human_fr, "vs_text": all_bleus_vs_text_fr},
        "en": {"vs_human": all_bleus_vs_human_en, "vs_text": all_bleus_vs_text_en},
    },
    "rouge1": {
        "fr": {"vs_human": rouge1_vs_human_fr, "vs_text": rouge1_vs_text_fr},
        "en": {"vs_human": rouge1_vs_human_en, "vs_text": rouge1_vs_text_en},
    },
    "rouge2": {
        "fr": {"vs_human": rouge2_vs_human_fr, "vs_text": rouge2_vs_text_fr},
        "en": {"vs_human": rouge2_vs_human_en, "vs_text": rouge2_vs_text_en},
    },
    "rougeL": {
        "fr": {"vs_human": rougeL_vs_human_fr, "vs_text": rougeL_vs_text_fr},
        "en": {"vs_human": rougeL_vs_human_en, "vs_text": rougeL_vs_text_en},
    },
}

with open("all_bleu_rouge_scores.json", "w") as f:
    json.dump(data, f, indent=4)

# saving correlations

pearson_bleus_vs_human = pearsonr(
    all_bleus_vs_human_fr, all_bleus_vs_human_en
).statistic
pearson_bleus_vs_text = pearsonr(all_bleus_vs_text_fr, all_bleus_vs_text_en).statistic

pearson_rouge1_vs_human = pearsonr(rouge1_vs_human_fr, rouge1_vs_human_en).statistic
pearson_rouge1_vs_text = pearsonr(rouge1_vs_text_fr, rouge1_vs_text_en).statistic

pearson_rouge2_vs_human = pearsonr(rouge2_vs_human_fr, rouge2_vs_human_en).statistic
pearson_rouge2_vs_text = pearsonr(rouge2_vs_text_fr, rouge2_vs_text_en).statistic

pearson_rougeL_vs_human = pearsonr(rougeL_vs_human_fr, rougeL_vs_human_en).statistic
pearson_rougeL_vs_text = pearsonr(rougeL_vs_text_fr, rougeL_vs_text_en).statistic

# save all the pearson correlations in a json file

pearson_correlations = {
    "bleu": {"vs_human": pearson_bleus_vs_human, "vs_text": pearson_bleus_vs_text},
    "rouge1": {"vs_human": pearson_rouge1_vs_human, "vs_text": pearson_rouge1_vs_text},
    "rouge2": {"vs_human": pearson_rouge2_vs_human, "vs_text": pearson_rouge2_vs_text},
    "rougeL": {"vs_human": pearson_rougeL_vs_human, "vs_text": pearson_rougeL_vs_text},
}

with open("all_pearson_correlations.json", "w") as f:
    json.dump(pearson_correlations, f, indent=4)

# save figures

plt.figure(figsize=(10, 10))
plt.scatter(all_bleus_vs_human_fr, all_bleus_vs_human_en)
plt.xlabel("BLEU score French")
plt.ylabel("BLEU score English")
plt.title("BLEU score of machine summaries vs human summaries in French and English")
plt.savefig("rouge_bleu_summeval_figures/bleu_vs_human.png")

plt.figure(figsize=(10, 10))
plt.scatter(all_bleus_vs_text_fr, all_bleus_vs_text_en)
plt.xlabel("BLEU score French")
plt.ylabel("BLEU score English")
plt.title("BLEU score of machine summaries vs text in French and English")
plt.savefig("rouge_bleu_summeval_figures/bleu_vs_text.png")

plt.figure(figsize=(10, 10))
plt.scatter(rouge1_vs_human_fr, rouge1_vs_human_en)
plt.xlabel("Rouge1 score French")
plt.ylabel("Rouge1 score English")
plt.title("Rouge1 score of machine summaries vs human summaries in French and English")
plt.savefig("rouge_bleu_summeval_figures/rouge1_vs_human.png")

plt.figure(figsize=(10, 10))
plt.scatter(rouge1_vs_text_fr, rouge1_vs_text_en)
plt.xlabel("Rouge1 score French")
plt.ylabel("Rouge1 score English")
plt.title("Rouge1 score of machine summaries vs text in French and English")
plt.savefig("rouge_bleu_summeval_figures/rouge1_vs_text.png")

plt.figure(figsize=(10, 10))
plt.scatter(rouge2_vs_human_fr, rouge2_vs_human_en)
plt.xlabel("Rouge2 score French")
plt.ylabel("Rouge2 score English")
plt.title("Rouge2 score of machine summaries vs human summaries in French and English")
plt.savefig("rouge_bleu_summeval_figures/rouge2_vs_human.png")

plt.figure(figsize=(10, 10))
plt.scatter(rouge2_vs_text_fr, rouge2_vs_text_en)
plt.xlabel("Rouge2 score French")
plt.ylabel("Rouge2 score English")
plt.title("Rouge2 score of machine summaries vs text in French and English")
plt.savefig("rouge_bleu_summeval_figures/rouge2_vs_text.png")

plt.figure(figsize=(10, 10))
plt.scatter(rougeL_vs_human_fr, rougeL_vs_human_en)
plt.xlabel("RougeL score French")
plt.ylabel("RougeL score English")
plt.title("RougeL score of machine summaries vs human summaries in French and English")
plt.savefig("rouge_bleu_summeval_figures/rougeL_vs_human.png")

plt.figure(figsize=(10, 10))
plt.scatter(rougeL_vs_text_fr, rougeL_vs_text_en)
plt.xlabel("RougeL score French")
plt.ylabel("RougeL score English")
plt.title("RougeL score of machine summaries vs text in French and English")
plt.savefig("rouge_bleu_summeval_figures/rougeL_vs_text.png")
