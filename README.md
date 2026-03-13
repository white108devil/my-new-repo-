# Bank Transaction Anamoly Detection (NNG-Mix, Semi-supervised)

This project implements a semi-supervised fraud/anomaly detection pipeline for
the Kaggle "Credit Card Transactions Dataset" (priyamchoksi). It uses NNG-Mix
(Nearest-Neighbor Gaussian Mixup) to generate pseudo anomalies, trains a model,
produces a colorful scatter plot, and exports a CSV containing detected fraud.

## Folder Structure

```
bank transaction anamoly detection/
  config.yaml
  requirements.txt
  data/
    raw/                # YOU place dataset here
    processed/          # auto-created
    output/             # auto-created
  models/               # auto-created
  outputs/
    plots/              # auto-created
  src/
    __init__.py
    data_utils.py
    nng_mix.py
    pipeline.py
    visualization.py
```

## Where to Put the Dataset

Put the CSV here (filename must match):

`data/raw/credit_card_transactions.csv`

If your filename is different, edit `config.yaml` -> `data.raw_path`.

## Step-by-step Execution

1. Open a terminal and go to the project folder:

```bash
cd "C:\Users\bhanu\OneDrive\Documents\bank transaction anamoly detection"
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the pipeline:

```bash
python src/pipeline.py --config config.yaml
```

## Outputs

- Fraud transactions CSV:
  `data/output/fraud_transactions.csv`
- Full predictions with scores:
  `data/output/anomaly_predictions.csv`
- Scatter plot:
  `outputs/plots/scatter_nng_mix.png`
- Confusion matrix:
  `outputs/plots/confusion_matrix.png`
- ROC curve:
  `outputs/plots/roc_curve.png`
- Score distribution:
  `outputs/plots/score_distribution.png`
- Pseudo anomalies (optional):
  `data/output/pseudo_anomalies.csv`

## Dataset-Specific Notes

The dataset includes numeric and categorical fields such as transaction time,
merchant, category, city/state, job, latitude/longitude, amount, and a fraud
label `is_fraud`. The pipeline automatically:
- extracts time features from `trans_date_trans_time`
- converts `dob` into `cardholder_age`
- frequency-encodes categorical columns to control memory usage
- drops high-cardinality identifiers like `cc_num` and `trans_num`
