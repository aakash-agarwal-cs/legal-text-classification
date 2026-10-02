# Classification of Legal Text (UE24CS352A ML Mini-Project)

Classifies US Supreme Court opinions into 15 issue areas (e.g. Criminal Procedure,
Civil Rights, Privacy, Economic Activity), based on the project "Classification of
Legal Text" (Iyer).

Team: Aakash Agarwal, Aaruni Choudhary

## Dataset
Supreme Court Database via the `textacy` package (~8,400 opinions with expert-assigned
`issue_area` labels). Unlabeled cases and the single-case class are dropped.
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
python baseline.py     # TF-IDF + logistic regression
python compare.py      # TF-IDF vs LDA vs Doc2Vec, writes results.csv
```

## Files
- `common.py`   shared data loading and fixed train/test split
- `baseline.py` TF-IDF baseline
- `compare.py`  method comparison

## Results
(fill in from results.csv)
