from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DefaultDataCollator
import pandas as pd
import numpy as np
import evaluate


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

train_ds = Dataset.from_pandas(pd.read_csv("data/train.csv").dropna())
validation_ds = Dataset.from_pandas(pd.read_csv("data/valid.csv").dropna())
test_ds = Dataset.from_pandas(pd.read_csv("data/test.csv").dropna())

label2id = {"E": 0, "I": 1, "A": 2, "O": 3}
id2label = {v: k for k, v in label2id.items()}

# Encode labels
train_ds = train_ds.map(lambda x: {"label": label2id[x["labels"]]})
train_ds = train_ds.remove_columns("labels")

validation_ds = validation_ds.map(lambda x: {"label": label2id[x["labels"]]})
validation_ds = validation_ds.remove_columns("labels")

test_ds = test_ds.map(lambda x: {"label": label2id[x["labels"]]})
test_ds = test_ds.remove_columns("labels")


def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

train_ds = train_ds.map(tokenize)
validation_ds = validation_ds.map(tokenize)
test_ds = test_ds.map(tokenize)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
validation_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=4)
data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="distilbert-tox-detection",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
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
