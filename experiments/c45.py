import os
import sys
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def read_file(path):
    if path.endswith("txt"):
        return load_from_txt(path)
    return load_data_from_parquet(path)

def load_data_from_parquet(path):
    df = pd.read_parquet(path)
    X = df[df.columns[1:]].values.astype(float)
    y = df[df.columns[0]].values.astype(float)
    return X, y


def load_from_txt(path):
    dataset = np.genfromtxt(path, delimiter=" ")
    X, y = dataset[:, 1:], dataset[:, 0]
    return X, y



datasets = sys.argv[1]
directory = sys.argv[2]
result_dir = f"{directory}/c45"

os.makedirs(f"{result_dir}", exist_ok=True)
res = list()

for file in os.listdir(datasets):
    if not file.endswith(".txt"):
        continue
    path = os.path.join(datasets, file)
    X, y = read_file(path)
    for depth in range(2, 9):
        clf = DecisionTreeClassifier(criterion="log_loss", max_depth=depth)
        start = time.perf_counter()
        clf.fit(X, y)
        runtime = time.perf_counter() - start
        out = {
            "name": Path(path).stem,
            "approach": "c4.5",
            "depth": depth,
            "runtime": runtime,
            "error": int((1 - clf.score(X, y)) * len(X))
        }
        res.append(out)

df = pd.DataFrame(res)
df.to_csv(f"{result_dir}/compiled.csv", index=None)
