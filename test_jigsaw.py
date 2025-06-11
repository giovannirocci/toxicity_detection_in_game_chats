from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import Dataset
import pandas as pd
import evaluate, argparse


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, type=str, default="distilbert-tox-detection")
parser.add_argument("-t", "--tokenizer", required=True, type=str, default="distilbert/distilbert-base-uncased")
parser.add_argument("-b", "--binary_model", action="store_true", default=False)
args = parser.parse_args()


model = AutoModelForSequenceClassification.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

jigsaw_df = pd.read_csv("data/jigsaw_toxicity/jigsaw_test.csv").dropna()
jigsaw_labels = pd.read_csv("data/jigsaw_toxicity/test_labels.csv").dropna()
test_df = jigsaw_df.merge(jigsaw_labels, on="id", how="inner")
test_df = test_df.drop(columns=["id", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
test_df = test_df.rename(columns={"comment_text": "text"})
test_df = test_df[test_df["toxic"] != -1]

label_conversion = {1: "toxic", 0: "neutral"}
test_df["labels"] = test_df["toxic"].apply(lambda x: label_conversion[x])
test_ds = Dataset.from_pandas(test_df)

binary_label2id = {"neutral": 0, "toxic": 1}
label2id = {"E": 0, "I": 1, "A": 2, "O": 3}

multiclass_to_binary = {
    0: 1,
    1: 1,
    2: 0,
    3: 0
}

test_ds = test_ds.map(lambda x: tokenizer(x["text"], padding=True, truncation=True, max_length=512), batched=True)
test_ds = test_ds.map(lambda x: {"label": binary_label2id[x["labels"]]})

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, batch_size=32)

references = test_ds["label"]
predictions = []

texts = test_ds["text"]
for out in pipe(texts, truncation=True, max_length=512):
    predictions.append(out["label"])


predictions = list(map(lambda x: binary_label2id[x] if args.binary_model else label2id[x], predictions))

if not args.binary_model:
    binary_preds = [multiclass_to_binary[p] for p in predictions]
else:
    binary_preds = predictions

metrics = evaluate.combine(["precision", "recall", "f1"])
result = metrics.compute(
    predictions=binary_preds,
    references=references,
    average="macro"
)

evaluate.save("./eval/jigsaw_eval/", **result)
