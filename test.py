from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import Dataset
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str, required=True)
parser.add_argument("--tokenizer", "-t", type=str, default="distilbert-base-uncased")
parser.add_argument("--test_dataset", "-td", type=str, required=True)
parser.add_argument("--output_file", "-o", type=str)
args = parser.parse_args()

model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


test_ds = Dataset.from_pandas(pd.read_csv(args.test_dataset).dropna())
test_ds = test_ds.map(lambda x: tokenizer(x["text"], padding=True, truncation=True, max_length=512), batched=True)
test_ds = test_ds.map(lambda x: {"label": {"E":0,"I":1,"A":2,"O":3}[x["labels"]]})

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

for utterance in test_ds:
    print(utterance)
    result = pipe(utterance["text"])
    print(result)
    if args.output_file:
        with open(args.output_file, "a") as f:
            f.write(f"{utterance['text']},{result[0]['label']}\n")
