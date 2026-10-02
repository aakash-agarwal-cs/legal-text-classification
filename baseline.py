"""
Step 1: load the Supreme Court dataset and train a baseline classifier.

Setup:
    pip install textacy scikit-learn pandas

Run:
    python baseline.py
"""
import pandas as pd
import textacy.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# 1. Load data (downloads ~ a few hundred MB the first time)
# ---------------------------------------------------------------
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

# Drop cases with no label or empty text
df = df.dropna(subset=["issue_area", "text"])
df = df[df["text"].str.len() > 0]

# -1 means "unlabeled", so drop it
df = df[df["issue_area"] != -1]

# Stratified split needs at least 2 examples per class
counts = df["issue_area"].value_counts()
df = df[df["issue_area"].isin(counts[counts >= 2].index)].reset_index(drop=True)

print("Total cases:", len(df))
print("\nIssue area distribution:")
print(df["issue_area"].value_counts().sort_index())

# ---------------------------------------------------------------
# 2. Stratified train/test split (keeps class proportions)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["issue_area"],
    test_size=0.2,
    random_state=42,
    stratify=df["issue_area"],
)

# ---------------------------------------------------------------
# 3. TF-IDF features
# ---------------------------------------------------------------
vec = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

# ---------------------------------------------------------------
# 4. Logistic regression
# ---------------------------------------------------------------
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(Xtr, y_train)
pred = clf.predict(Xte)

print("\nAccuracy :", round(accuracy_score(y_test, pred), 3))
print("Macro-F1 :", round(f1_score(y_test, pred, average="macro"), 3))
print("\nPer-class report:")
print(classification_report(y_test, pred, zero_division=0))
