import pandas as pd
import torch, evaluate
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset

from test import hyperparams

# Load the model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')

test_df = pd.read_csv('data/test.csv')
binary_labels = {"E": "toxic", "I": "toxic", "A": "neutral", "O": "neutral"}

label2id = {"toxic": 0, "neutral": 1}

test_df["labels"] = test_df["labels"].apply(lambda x: binary_labels[x] if x in binary_labels else "N")
test_ds = Dataset.from_pandas(test_df.dropna())

test_ds = test_ds.map(lambda x: tokenizer(x["text"], padding=True, truncation=True, max_length=512), batched=True)
test_ds = test_ds.map(lambda x: {"label": label2id[x["labels"]]})

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

references = test_ds["label"]
predictions = []
for out in pipe(KeyDataset(test_ds, "text")):
    predictions.append(out["label"])


predictions = list(map(lambda x: label2id[x], predictions))

metrics = evaluate.combine(["precision", "recall", "f1"])
result = metrics.compute(
    predictions=predictions,
    references=references,
    average="macro"
)

evaluate.save("./eval/", **result)