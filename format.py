from csv import DictReader

FLAG_DICT = {flag: f"__label__{flag}" for flag in ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")}

false_newline = "\\n"

p = True
with open("train.csv", "r", encoding="utf-8") as csvfile:
    csv_data = csvfile
    reader = [i for i in DictReader(csv_data)]

with open("labeled.data", "w", encoding="utf-8", newline="") as f:
    for row in reader:
        labels = (FLAG_DICT[flag] for flag in FLAG_DICT if row[flag] == "1")
        text = row["comment_text"].replace("\n", false_newline)

        line = f"{' '.join(labels) or '__label__non_toxic'} {text}".replace('"', "")
        f.write(line + '\n')