"""
Compares three ways of turning a legal opinion into features, all followed by
the same logistic regression classifier on the same train/test split:

    1. TF-IDF            (baseline)
    2. LDA topics        (paper's approach 1)
    3. Doc2Vec vectors   (paper's approach 2)

Setup:   pip install gensim   (on top of the packages from baseline.py)
Run:     python compare.py
Output:  results.csv  (accuracy and macro-F1 per method)
"""
import time

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from common import load_split

# ---- settings you can tweak (lower these if it runs too slowly) ----
NUM_TOPICS = 50
LDA_PASSES = 5
D2V_DIM = 100
D2V_EPOCHS = 10
# --------------------------------------------------------------------


def tokenize(text):
    return [
        t
        for t in simple_preprocess(text, min_len=3, max_len=20)
        if t not in STOPWORDS
    ]


def evaluate(name, Xtr, Xte, y_train, y_test, results):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.3f}   Macro-F1: {f1:.3f}")
    print(classification_report(y_test, pred, zero_division=0))
    results.append({"model": name, "accuracy": round(acc, 3), "macro_f1": round(f1, 3)})


def lda_features(lda, bows, k):
    X = np.zeros((len(bows), k))
    for i, bow in enumerate(bows):
        for topic_id, prob in lda.get_document_topics(bow, minimum_probability=0.0):
            X[i, topic_id] = prob
    return X


def main():
    t0 = time.time()
    print("Loading data and split...")
    X_train, X_test, y_train, y_test = load_split()
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    results = []

    # ---------------- 1. TF-IDF ----------------
    print("\n[1/3] TF-IDF + LR")
    vec = TfidfVectorizer(
        stop_words="english", max_features=50000, min_df=3, max_df=0.9, sublinear_tf=True
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    evaluate("TF-IDF + LR", Xtr, Xte, y_train, y_test, results)

    # Tokenize once, reuse for LDA and Doc2Vec
    print("\nTokenizing documents (takes a minute)...")
    train_tokens = [tokenize(t) for t in X_train]
    test_tokens = [tokenize(t) for t in X_test]

    # ---------------- 2. LDA ----------------
    print(f"\n[2/3] LDA ({NUM_TOPICS} topics) + LR")
    dictionary = Dictionary(train_tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=50000)
    train_bows = [dictionary.doc2bow(t) for t in train_tokens]
    test_bows = [dictionary.doc2bow(t) for t in test_tokens]

    lda = LdaMulticore(
        corpus=train_bows,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=LDA_PASSES,
        workers=3,
        random_state=42,
    )
    Xtr = lda_features(lda, train_bows, NUM_TOPICS)
    Xte = lda_features(lda, test_bows, NUM_TOPICS)
    evaluate(f"LDA ({NUM_TOPICS} topics) + LR", Xtr, Xte, y_train, y_test, results)

    # Print the topics, useful for the write-up
    print("\nSample LDA topics:")
    for tid, words in lda.show_topics(num_topics=8, num_words=8, formatted=False):
        print(f"  Topic {tid}: {', '.join(w for w, _ in words)}")

    # ---------------- 3. Doc2Vec ----------------
    print(f"\n[3/3] Doc2Vec ({D2V_DIM}-dim) + LR")
    tagged = [TaggedDocument(toks, [i]) for i, toks in enumerate(train_tokens)]
    d2v = Doc2Vec(vector_size=D2V_DIM, min_count=5, epochs=D2V_EPOCHS, workers=4, seed=42)
    d2v.build_vocab(tagged)
    d2v.train(tagged, total_examples=d2v.corpus_count, epochs=d2v.epochs)

    Xtr = np.vstack([d2v.dv[i] for i in range(len(train_tokens))])
    print("Inferring vectors for test documents...")
    Xte = np.vstack([d2v.infer_vector(t) for t in test_tokens])
    evaluate(f"Doc2Vec ({D2V_DIM}d) + LR", Xtr, Xte, y_train, y_test, results)

    # ---------------- Summary ----------------
    table = pd.DataFrame(results)
    print("\n===== SUMMARY =====")
    print(table.to_string(index=False))
    table.to_csv("results.csv", index=False)
    print(f"\nSaved results.csv  (total time {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":  # required on Windows (LdaMulticore uses multiprocessing)
    main()
