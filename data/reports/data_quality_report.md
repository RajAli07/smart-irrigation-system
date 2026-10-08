# SC07 Data Quality Report

## Purpose

This report records how the Smart Irrigation dataset was inspected, cleaned, validated, and split for the SC07 project[cite: 55].

## Dataset Source

* **Dataset:** Irrigation Water Requirement Prediction Dataset[cite: 55]
* **Source:** Kaggle
* **Original records:** 10,000+
* **Target column:** Irrigation_Need[cite: 55]

## Cleaning Method

* The original raw CSV was not modified[cite: 55].
* Missing numeric values were filled using the median of their own column[cite: 55].
* Exact duplicate rows were removed[cite: 55].

## Reproducible Split

A fixed random seed of 42 was used. The data was split separately to preserve the balanced class distribution[cite: 56].

| Dataset | Rows | Purpose |
| :--- | :--- | :--- |
| `train.csv` | 7,000 | Model training[cite: 56] |
| `validation.csv` | 1,500 | Model selection and tuning[cite: 56] |
| `test.csv` | 1,500 | Final unseen evaluation[cite: 56] |
| **Total** | **10,000** | **Complete cleaned dataset**[cite: 56] |

## Output Files

The pipeline created these files in `data/processed/`[cite: 56]:

* `irrigation_clean.csv`
* `train.csv`
* `validation.csv`
* `test.csv`
