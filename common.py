"""Shared data loading so every model uses the exact same train/test split."""
import pandas as pd
import textacy.datasets
from sklearn.model_selection import train_test_split

SEED = 42


def load_data() -> pd.DataFrame:
    ds = textacy.datasets.SupremeCourt()
    ds.download()

    rows = []
    for text, meta in ds.records():
        rows.append(
            {
                "text": text,
                "issue": meta.get("issue"),
                "issue_area": meta.get("issue_area"),
            }
        )
    df = pd.DataFrame(rows)

    df = df.dropna(subset=["issue_area", "text"])
    df = df[df["text"].str.len() > 0]

    # -1 means "unlabeled", so drop it
    df = df[df["issue_area"] != -1]

    # Stratified split needs at least 2 examples per class
    counts = df["issue_area"].value_counts()
    df = df[df["issue_area"].isin(counts[counts >= 2].index)]
    return df.reset_index(drop=True)


def load_split(test_size: float = 0.2):
    """Returns X_train, X_test (lists of str) and y_train, y_test (lists of int)."""
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["issue_area"],
        test_size=test_size,
        random_state=SEED,
        stratify=df["issue_area"],
    )
    return (
        X_train.tolist(),
        X_test.tolist(),
        y_train.tolist(),
        y_test.tolist(),
    )
