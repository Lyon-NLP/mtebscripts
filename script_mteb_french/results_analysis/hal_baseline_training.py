import argparse

import numpy as np

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

DATASET = "lyon-nlp/clustering-hal-s2s"
MODEL = "almanach/camembert-base"
SEED = 42


def main(args):
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    dataset = load_dataset(args.dataset, split="test")
    dataset = dataset.rename_columns({
        "title": "text",
        "domain": "label"
    }).remove_columns(["hal_id"])

    dataset = dataset.filter(lambda example: example["label"] not in ("image"))
    print(dataset)

    dataset = dataset.class_encode_column("label")
    
    dataset = dataset.train_test_split(test_size=0.3, shuffle=True, stratify_by_column="label", seed=args.seed)
    dataset_ = dataset["train"].train_test_split(test_size=0.1, shuffle=True, stratify_by_column="label", seed=args.seed)
    dataset["train"] = dataset_["train"]
    dataset["validation"] = dataset_["test"]
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, 
        num_labels=dataset["train"].features["label"].num_classes
    ).to(device)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="macro")

    metric = evaluate.load("f1")

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        output_dir=args.output_dir,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    score = trainer.evaluate(
        eval_dataset=dataset["test"]
    ) 
    print(score)

    trainer.save_model(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="hal_baseline")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
