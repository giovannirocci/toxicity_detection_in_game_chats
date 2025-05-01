from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
import pandas as pd
import argparse, evaluate


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str, required=True)
parser.add_argument("--tokenizer", "-t", type=str, default="distilbert-base-uncased")
parser.add_argument("--test_dataset", "-td", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default="output")
parser.add_argument("--single_class", "-sc", action="store_true", default=False)
args = parser.parse_args()

model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

label2id = {"E": 0, "I": 1, "A": 2, "O": 3}

test_ds = Dataset.from_pandas(pd.read_csv(args.test_dataset).dropna())
test_ds = test_ds.map(lambda x: tokenizer(x["text"], padding=True, truncation=True, max_length=512), batched=True)
test_ds = test_ds.map(lambda x: {"label": label2id[x["labels"]]})

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

references = test_ds["label"]
predictions = []
for out in pipe(KeyDataset(test_ds, "text")):
    predictions.append(out["label"])

predictions = list(map(lambda x: label2id[x], predictions))

clf_metrics = evaluate.combine(["precision", "recall", "f1"])
result = clf_metrics.compute(
    predictions=predictions,
    references=references,
    labels=list(label2id.values()) if args.single_class else None,
    average=None if args.single_class else "macro")

# Convert numpy arrays to lists (for JSON serialization)
if args.single_class:
    for k in result:
        if hasattr(result[k], 'tolist'):
            result[k] = result[k].tolist()

hyperparams = {"model": args.model_name, "tokenizer": args.tokenizer}
evaluate.save(f"./{args.output_dir}/", **result, **hyperparams)

