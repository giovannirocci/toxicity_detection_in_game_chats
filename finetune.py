from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DefaultDataCollator
import pandas as pd
import numpy as np
import evaluate, argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train_file", "-tf", type=str, default="data/train.csv")
parser.add_argument("--valid_file", "-vf", type=str, default="data/valid.csv")
parser.add_argument("--model_name", "-m", type=str, default="distilbert/distilbert-base-uncased")
parser.add_argument("--num_categories", "-nc", type=int, default=4)
parser.add_argument("--output_dir", "-o", type=str, default="output")
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_name)

train_ds = Dataset.from_pandas(pd.read_csv(args.train_file).dropna())
validation_ds = Dataset.from_pandas(pd.read_csv(args.valid_file).dropna())

label2id = {"E": 0, "I": 1, "A": 2, "O": 3}
id2label = {v: k for k, v in label2id.items()}

# Encode labels
train_ds = train_ds.map(lambda x: {"label": label2id[x["labels"]]})
train_ds = train_ds.remove_columns("labels")

validation_ds = validation_ds.map(lambda x: {"label": label2id[x["labels"]]})
validation_ds = validation_ds.remove_columns("labels")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

train_ds = train_ds.map(tokenize)
validation_ds = validation_ds.map(tokenize)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
validation_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_categories)
data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir=args.output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    label_names=list(label2id.keys()),
)

metric = evaluate.load("f1")

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=validation_ds,
    compute_metrics=compute_metrics
)

trainer.train()
