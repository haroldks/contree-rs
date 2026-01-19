import sys
import os
import json
import pandas as pd
from pycontree import ConTree
from sklearn.model_selection import cross_validate, KFold


def main():
    # 1. Capture Command Line Arguments
    if len(sys.argv) != 4:
        print("Usage: python run.py <dataset_path> <max_depth> <output_dir>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    depth = int(sys.argv[2])
    output_file = sys.argv[3]

    # 2. Load Data
    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} not found.")
        sys.exit(1)

    df = pd.read_csv(dataset_path, sep=" ", header=None)
    X = df[df.columns[1:]]
    y = df[0]

    # 3. Define Models
    models = {
        "ConTree": ConTree(max_depth=depth)
    }
    n_splits = 5
    results_summary = {
        "metadata": {
            "dataset": dataset_path,
            "max_depth": depth,
            "folds": n_splits
        },
        "results": {}
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 4. Evaluation
    for name, model in models.items():
        cv_results = cross_validate(
            model, X, y,
            cv=kf,
            scoring='accuracy',
            return_train_score=True
        )

        # Store both summaries and raw fold-by-fold lists
        results_summary["results"][name] = {
            "summary": {
                "mean_test_acc": round(cv_results['test_score'].mean(), 4),
                "std_test_acc": round(cv_results['test_score'].std(), 4),
                "mean_train_acc": round(cv_results['train_score'].mean(), 4)
            },
            "folds": {
                "test_scores": cv_results['test_score'].tolist(),
                "train_scores": cv_results['train_score'].tolist()
            }
        }



    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=4)


if __name__ == "__main__":
    main()
