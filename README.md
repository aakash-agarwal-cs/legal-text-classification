# Classification of Legal Text (UE24CS352A ML Mini-Project)

Predicts the issue area (e.g. Criminal Procedure, Civil Rights, Privacy, Economic Activity)
of a US Supreme Court opinion from its full text, based on the project "Classification of
Legal Text" (Iyer). Three feature methods (TF-IDF, LDA topics, Doc2Vec) are compared with the
same logistic regression classifier on the same train/test split, and the best one (TF-IDF)
powers a small demo app.

Team: Aakash Agarwal, Aaruni Choudhary

## Dataset
Supreme Court Database via the `textacy` package (8,417 opinions with expert-assigned
`issue_area` labels). 23 unlabeled cases and the single-case class 14 are dropped, leaving
**8,393 opinions in 13 issue areas**. Stratified 80/20 split (seed 42): 6,714 train / 1,679 test.
The dataset downloads automatically on first run.

## Setup
Requires **Python 3.12** (3.13 fails to install `textacy`'s dependency `floret` on Windows).

```
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

## Run
```
python baseline.py       # TF-IDF + logistic regression
python compare.py        # TF-IDF vs LDA vs Doc2Vec, writes results.csv (takes ~15-30 min)
```

### Demo app
```
python train_model.py    # trains the final model, saves model/model.joblib (run once)
streamlit run app.py     # opens the web app: paste an opinion, get predicted topics + similar cases
```
`model/` is not stored in the repo (the file is large), so run `train_model.py` first.

## Files
- `common.py`   shared data loading, cleaning and fixed train/test split
- `baseline.py` TF-IDF + logistic regression baseline
- `compare.py`  comparison of TF-IDF, LDA and Doc2Vec
- `train_model.py` trains and saves the final TF-IDF model for the app
- `app.py`      Streamlit demo app
- `results.csv` output of `compare.py`

## Results
Test set of 1,679 opinions, 13 issue areas.

| Method | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + LR | 0.777 | 0.738 |
| LDA (50 topics) + LR | 0.600 | 0.504 |
| Doc2Vec (100-d) + LR | 0.634 | 0.521 |

TF-IDF is clearly the best. The classes are heavily imbalanced, so macro-F1 is reported
alongside accuracy.
