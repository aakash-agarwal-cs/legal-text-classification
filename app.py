"""
Streamlit demo: paste a court case, get the top 3 predicted legal topics
and the 3 most similar cases from the dataset.

Prerequisite:  python train_model.py   (creates model/model.joblib)
Run:           streamlit run app.py
"""
import os

import joblib
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")

st.set_page_config(page_title="Legal Text Classifier", page_icon="⚖️")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


st.title("⚖️ Supreme Court Case Classifier")
st.caption("Paste the text of a US Supreme Court opinion to predict its legal topic.")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"`{MODEL_PATH}` not found. Run `python train_model.py` first, "
        "then restart the app."
    )
    st.stop()

bundle = load_bundle()
vec = bundle["vectorizer"]
clf = bundle["classifier"]
doc_matrix = bundle["doc_matrix"]
case_names = bundle["case_names"]
us_cites = bundle["us_cites"]
issue_areas = bundle["issue_areas"]
labels = bundle["labels"]

text = st.text_area("Case text", height=300, placeholder="Paste the opinion text here...")

if st.button("Predict", type="primary"):
    if len(text.strip()) < 50:
        st.warning("Please paste a longer piece of text (at least a few sentences).")
        st.stop()

    X = vec.transform([text])
    if X.nnz == 0:
        st.warning("None of the words in this text are in the model's vocabulary.")
        st.stop()

    # ---- Top 3 topics ----
    proba = clf.predict_proba(X)[0]
    top = np.argsort(proba)[::-1][:3]

    st.subheader("Top 3 predicted topics")
    for i in top:
        name = labels.get(int(clf.classes_[i]), f"Area {clf.classes_[i]}")
        pct = proba[i] * 100
        st.write(f"**{name}: {pct:.0f}%**")
        st.progress(float(proba[i]))

    # ---- 3 most similar cases ----
    # TF-IDF vectors are L2-normalised, so the dot product is cosine similarity.
    sims = (doc_matrix @ X.T).toarray().ravel()
    order = np.argsort(sims)[::-1]
    # Skip the case itself if the pasted text is already in the dataset.
    order = [j for j in order if sims[j] < 0.999][:3]

    st.subheader("3 most similar cases")
    for j in order:
        cite = f" ({us_cites[j]})" if us_cites[j] else ""
        topic = labels.get(int(issue_areas[j]), "Unknown")
        st.write(f"**{case_names[j]}**{cite}")
        st.caption(f"Topic: {topic} · Similarity: {sims[j] * 100:.0f}%")