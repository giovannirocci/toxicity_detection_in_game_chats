from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

model = AutoModelForSequenceClassification.from_pretrained("distilbert-tox-detection/best_model")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# If needed, reload test dataset
from datasets import Dataset
import pandas as pd

test_ds = Dataset.from_pandas(pd.read_csv("data/test.csv").dropna())
test_ds = test_ds.map(lambda x: tokenizer(x["text"], padding=True, truncation=True, max_length=512), batched=True)
test_ds = test_ds.map(lambda x: {"label": {"E":0,"I":1,"A":2,"O":3}[x["labels"]]})
test_ds = test_ds.remove_columns("labels")
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Then reuse Trainer
trainer = Trainer(model=model)

test_results = trainer.predict(test_ds)
print(test_results.metrics)
