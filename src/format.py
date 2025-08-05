from csv import DictReader

FLAG_DICT = {flag: f"__label__{flag}" for flag in ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")}

false_newline = "\\n"

def format_data(input: str = "train.csv", output: str = "labeled.data"):
    print(f"formatting data in {input}...")
    with open(input, "r", encoding="utf-8") as csvfile, open(output, "w", encoding="utf-8", newline="") as labeled:
        for row in DictReader(csvfile):
            labels = (FLAG_DICT[flag] for flag in FLAG_DICT if row[flag] == "1")
            text = row["comment_text"].replace("\n", false_newline)

            line = f"{' '.join(labels) or '__label__non_toxic'} {text}".replace('"', "")
            labeled.write(line + '\n')
        print(f"labeled data output to {output}")
    print("success")
