# Predicting Rocket Launch Failures Using Machine Learning: A Study of Modern Orbital Launches

## Project Overview

This project focuses on predicting rocket launch failures using machine learning. By combining launch data from the Launch Library 2 API with historical weather data from the Open-Meteo API, several predictive models were developed to identify high-risk launch scenarios.

The project investigates whether launch characteristics, launch history, infrastructure maturity, and weather conditions can be used to predict failed orbital launches.

For the full methodology, results, and discussion, see the accompanying report: `project_report.pdf`.

---

## Project Pipeline

![Project Pipeline](images/project_pipeline.png)

The project follows the workflow below:

1. Collect launch data from Launch Library 2 API
2. Collect weather data from Open-Meteo API
3. Clean and preprocess the dataset
4. Perform feature engineering
5. Train machine learning models
6. Evaluate model performance
7. Analyze feature importance

---

## Key Results

The dataset contains orbital launches from 2014–2025 and combines launch-related variables with historical weather conditions.

Since launch failures represent only a small proportion of all launches (~5%), model performance was evaluated primarily using Recall, Precision, F1 Score, and Precision-Recall Area Under the Curve (PR-AUC).

| Model               | Recall   | Precision | F1 Score | PR-AUC   |
| ------------------- | -------- | --------- | -------- | -------- |
| Logistic Regression | 0.58     | 0.24      | 0.34     | 0.21     |
| Random Forest       | 0.74     | 0.23      | 0.35     | 0.28     |
| XGBoost             | **0.84** | 0.17      | 0.28     | **0.35** |

### Main Findings

* XGBoost achieved the highest recall and PR-AUC.
* The model successfully identified 84% of failed launches in the test set.
* Launch experience variables were among the strongest predictors.
* Weather variables also contributed to prediction performance.
* All models substantially outperformed a random baseline.

Because launch failures are safety-critical events, the project prioritizes identifying failures (high recall) rather than minimizing false alarms.

---

## Feature Importance

![XGBoost Feature Importance](images/xgboost_feature_importance.png)

The feature importance analysis indicates that variables related to launch pad experience, agency launch history, rocket maturity, and weather conditions contribute substantially to launch failure prediction.

Notably, operational experience variables consistently appeared among the most important predictors across models.

---

## Repository Structure

### 1. Data Collection, Cleaning & Machine Learning Analysis

* `data_utils.py`

  * Shared helper functions (saving/loading JSON progress files) used by both data collection scripts.

* `01_launch_data_collection.py`

  * Retrieves rocket launch data from the Launch Library 2 API.

* `02_weather_data_collection.py`

  * Retrieves weather data from the Open-Meteo API based on launch site coordinates.

* `03_data_cleaning.py`

  * Handles missing values and preprocessing.

* `04_feature_engineering_and_ML.ipynb`

  * Feature engineering, class imbalance handling, model training, evaluation, and feature importance analysis.

### 2. Demonstration Resources

`DEMO data collection/`

Contains simplified versions of the data collection notebooks for quickly testing API functionality without waiting for the complete data retrieval process.

* `01_launch_data_collection_DEMO.ipynb`
* `02_weather_data_collection_DEMO.ipynb`

---

## Requirements

Install the following Python packages:

```text
pandas
numpy
requests
retry-requests
scikit-learn
xgboost
matplotlib
seaborn
```

---

## How to Run

### Option 1: Full Procedure (01 → 04)

1. Clone the repository.
2. Run steps 1–3 from the command line, in order:

```bash
python 01_launch_data_collection.py
python 02_weather_data_collection.py
python 03_data_cleaning.py
```

3. Open and run `04_feature_engineering_and_ML.ipynb` in Jupyter.

**Important:**

The data collection scripts perform live API calls to:

* Launch Library 2 API
* Open-Meteo API

Due to API rate limits and the volume of historical data, the complete collection process may require a significant amount of time. Both scripts save progress as they go, so an interrupted run can simply be re-run to resume where it left off.

For quick testing, use the notebooks located in:

```text
DEMO data collection/
```

---

### Option 2: Data Analysis & Machine Learning Only

If you wish to skip data collection:

1. Ensure `launch_dataset.csv` is located in the repository root directory.
2. Run:

```bash
python 03_data_cleaning.py
```

3. Open and run `04_feature_engineering_and_ML.ipynb` in Jupyter.

These steps are pre-configured to use the full dataset already provided in the repository.

---

## Reproducibility

All results reported in the accompanying project report can be reproduced using the scripts and notebook provided in this repository.

The repository includes both the complete workflow (data collection through model evaluation) and a shortcut path for directly reproducing the machine learning results using the supplied dataset.
