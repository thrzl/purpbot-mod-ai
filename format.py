from pandas import read_csv
import numpy as np
import csv
from sys import stdout
from tqdm import tqdm

COLUMNS = ("id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate")
FLAGS = ("toxic","severe_toxic","obscene","threat","insult","identity_hate")
DATA_LIMIT = 75000

open('all_categories.data',"w",encoding='utf-8').close()

a = np.array(read_csv("train.csv"))

data = [{col: entry[i] for i, col in enumerate(COLUMNS)} for entry in a]
real_newline = "\n"
false_newline = "\\n"
t = "\n".join(f"{'toxic' if any(d[flag] for flag in FLAGS) else 'not_toxic'} {d['comment_text'].replace(real_newline,false_newline).lower()}" for d in data)

with open("all_categories.data", "a", encoding='utf-8') as f:
    for entry in tqdm(data[:DATA_LIMIT]):
        f.write(f"{' '.join('__label__'+flag for flag in FLAGS if entry[flag] == 1) or '__label__non_toxic'} {entry['comment_text'].replace(real_newline,false_newline).lower()}\n")

