import csv
from tqdm import tqdm

COLUMNS = (
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
FLAGS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")

false_newline = "\\n"

with open("all_categories.data", "w", encoding="utf-8", newline="") as f:
    with open("train.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader):
            labels = [f"__label__{flag}" for flag in FLAGS if row[flag] == "1"]
            if not labels:
                labels = ["__label__non_toxic"]
            text = row["comment_text"].replace("\n", false_newline)

            line = f"{' '.join(labels)} {text}"
            line = line.replace('"', "")

            f.write(line + "\n")
