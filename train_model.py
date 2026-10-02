"""
Trains the final TF-IDF + logistic regression model on all labeled cases and
saves everything the demo app needs into model/model.joblib.

Run once:   python train_model.py
"""
import os

import joblib
import numpy as np
import pandas as pd
import textacy.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELS = {
    1: "Criminal Procedure",
    2: "Civil Rights",
    3: "First Amendment",
    4: "Due Process",
    5: "Privacy",
    6: "Attorneys",
    7: "Unions",
    8: "Economic Activity",
    9: "Judicial Power",
    10: "Federalism",
    11: "Interstate Relations",
    12: "Federal Taxation",
    13: "Miscellaneous",
}

print("Loading data...")
ds = textacy.datasets.SupremeCourt()
ds.download()

rows = []
for text, meta in ds.records():
    rows.append(
        {
            "text": text,
            "issue_area": meta.get("issue_area"),
            "case_name": meta.get("case_name"),
            "us_cite": meta.get("us_cite"),
        }
    )
df = pd.DataFrame(rows)
df = df.dropna(subset=["issue_area", "text"])
df = df[(df["text"].str.len() > 0) & (df["issue_area"].isin(LABELS.keys()))]
df = df.reset_index(drop=True)
print("Cases:", len(df))

print("Fitting TF-IDF...")
vec = TfidfVectorizer(
    stop_words="english", max_features=50000, min_df=3, max_df=0.9, sublinear_tf=True
)
X = vec.fit_transform(df["text"]).astype(np.float32)

print("Training logistic regression...")
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X, df["issue_area"])

os.makedirs("model", exist_ok=True)
bundle = {
    "vectorizer": vec,
    "classifier": clf,
    "doc_matrix": X,  # used for similar-case search
    "case_names": df["case_name"].fillna("(unknown case)").tolist(),
    "us_cites": df["us_cite"].fillna("").tolist(),
    "issue_areas": df["issue_area"].tolist(),
    "labels": LABELS,
}
joblib.dump(bundle, "model/model.joblib")
print("Saved model/model.joblib")
