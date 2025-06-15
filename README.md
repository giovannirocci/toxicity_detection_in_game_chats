# Toxicity Detection in In-Game Chat

This repository contains code and resources for detecting toxicity in online multiplayer games using transformer-based models. It includes scripts for preprocessing data, fine-tuning DistilBERT models, testing them across different datasets, and visualizing evaluation metrics.

## Project Structure
```
.
├── data/                      # Directory for train/validation/test CSVs for different datasets
├── eval/                   # Directory for model outputs and evaluation results
├── plots/                   # Directory to save comparison visualizations
├── distilbert-tox-detection-no-context/     # Weights for the DistilBERT model trained on toxicity data without context
├── distilbert-tox-detection/  # Weights for the DistilBERT model trained on toxicity data with context
├── finetune.py               # Fine-tunes a DistilBERT model on toxic chat data
├── preprocess.py             # Processes raw datasets (adds chat context, filters labels)
├── test.py                   # Evaluates trained models on multi-class data
├── test_jigsaw.py            # Evaluates models on the binary Jigsaw toxicity dataset
├── roberta_test.py           # Evaluates a pretrained RoBERTa classifier on both CODNA and Jigsaw datasets
├── plot.py                   # Generates visualizations comparing model performance
├── requirements.txt          # Required Python packages

```

## Usage
### 1. Preprocessing Data
Execute the following command to prepare the data for fine-tuning and evaluation (with or without context):
```bash
python preprocess.py \
  --input_training data/CONDA_train.csv \
  --input_validation data/CONDA_valid.csv \
  --output_training data/train.csv \
  --output_validation data/valid.csv \
  --output_test data/test.csv \
  [--no_context]
```
Add the `--no_context` flag to preprocess the data without including chat context for each entry.

### 2. Fine-tuning a DistilBERT Model
Execute the following command to fine-tune a (DistilBERT) model on the processed data:
```bash
python finetune.py \
  --train_file data/train.csv \
  --validation_file data/valid.csv \
  --output_dir your_finetuned_model_name/ \
  [--model_name distilbert-base-uncased] \
  [--num_categories 4] 
```

### 3. Evaluating the Model
#### On Multi-Class Data (CONDA)
```bash
python test.py \
  --model_name your_finetuned_model_name/ \
  --test_file data/test.csv \
  --output_dir eval/ \
  [--single_class]
```
Add the `--single_class` flag to evaluate model performance for each label class separately.

#### On Binary Jigsaw Dataset
```bash
python test_jigsaw.py \
  --model your_finetuned_model_name/ \
  --tokenizer distilbert/distilbert-base-uncased \
  [--binary_model]
```
The script assumes by default that the model is trained on multi-class data. Use the `--binary_model` flag if you want to evaluate a model trained for binary classification.

### 4. Evaluating a Pretrained RoBERTa Classifier
```bash
python roberta_test.py 
```
This script will test a pretrained RoBERTa classifier on the CONDA dataset, outputting results to the `eval/` directory. To test it on the Jigsaw dataset, run the `test_jigsaw.py` script with the `--binary_model` flag instead.

### 5. Visualizing Results
```bash
python plot.py \
  --model1 eval/your_model_results.json \
  --model2 eval/your_other_model_results.json \
  [--model3 eval/your_third_model_results.json] \
  --output_dir plots/ 
```
The script will automatically handle both mean and per-class metrics, generating visualizations comparing the performance of the specified models.

## Requirements
Install the required packages using pip:
```bash
pip install -r requirements.txt
```