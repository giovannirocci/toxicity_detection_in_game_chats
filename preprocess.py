import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_training", "-itr", type=str)
parser.add_argument("--input_validation", "-iv", type=str)
parser.add_argument("--output_training", "-otr", type=str)
parser.add_argument("--output_validation", "-ov", type=str)
parser.add_argument("--output_test", "-ots", type=str)
args = parser.parse_args()

from sklearn.model_selection import train_test_split

train = pd.read_csv(args.input_training)
validation = pd.read_csv(args.input_validation)

def clean_utterances(df):
    cleaned_utterances = []
    for _, row in df.iterrows():
        row['utterance'] = str(row['utterance']).replace(" [SEPA] ", " ")
        cleaned_utterances.append(row['utterance'])
    return cleaned_utterances

def process(df, outfile):
    df['chatHistory'] = ""

    history_sep = " [SEP] "
    chat_history_column = []

    df['utterance'] = clean_utterances(df)

    for _, group in tqdm(df.groupby("conversationId")):
        chat_history = []
        for i, row in group.iterrows():
            # Save the current chat_history before adding the current line
            chat_history_column.append(history_sep.join(chat_history))
            chat_history.append(str(row["utterance"]))

    # Assign to the dataframe
    df["chatHistory"] = chat_history_column

    df["text"] = df["chatHistory"] + history_sep + df["utterance"]
    df["labels"] = df["intentClass"]

    df["text"] = [row["text"].lstrip(" [SEP] ") for _,row in df.iterrows()]

    df[["text", "labels"]].to_csv(outfile, index=False)


if __name__ == "__main__":
    # Combine original train and validation
    combined_df = pd.concat([train, validation], ignore_index=True)

    # Clean and filter valid labels
    combined_df = combined_df.dropna(subset=["utterance", "intentClass"])
    combined_df = combined_df[combined_df["intentClass"].isin(["E", "I", "A", "O"])]

    # Split: 60% train, 20% val, 20% test
    train_split, temp_split = train_test_split(
        combined_df,
        test_size=0.4,
        stratify=combined_df["intentClass"],
        random_state=42
    )

    valid_split, test_split = train_test_split(
        temp_split,
        test_size=0.5,
        stratify=temp_split["intentClass"],
        random_state=42
    )

    # Process and save
    process(train_split, args.output_training)
    process(valid_split, args.output_validation)
    process(test_split, args.output_test)

