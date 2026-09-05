# End-to-End MLOps CI/CD Pipeline

An end-to-end Machine Learning Operations (MLOps) project for predicting loan defaults and deploying the trained model as a containerized REST API.

The project demonstrates how a machine learning workflow can be taken from raw data ingestion through data transformation, model training, evaluation, MLflow model management, automated testing, containerization, and API-based inference.

## 🚀 Project Overview

The pipeline processes Lending Club loan data and builds a binary classification model to predict whether a loan is likely to default.

The project covers:

* Data ingestion and processing with **PySpark**
* Bronze → Silver → Gold data architecture
* Data validation and feature engineering
* Train/test splitting and sampling
* Logistic Regression model training
* Model evaluation and validation
* **MLflow** experiment tracking
* MLflow model registration and Champion promotion
* Automated testing with **pytest**
* CI using **GitHub Actions**
* Containerization with **Docker**
* REST API deployment using **FastAPI**
* Request validation using **Pydantic**
* Production-style model loading from the MLflow Champion alias

## 🏗️ End-to-End Architecture

```text
                    ┌──────────────────────┐
                    │   Raw Lending Data   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Bronze Ingestion   │
                    │       PySpark        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Silver Processing   │
                    │ Validation + Cleanup │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Gold Transformation │
                    │ Feature Engineering  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Train / Test Split   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Model Training     │
                    │ Logistic Regression  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      MLflow          │
                    │ Tracking + Registry  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Model Validation     │
                    │ & Promotion Rules    │
                    └──────────┬───────────┘
                               │
                         Champion Alias
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Docker         │
                    │   FastAPI Service    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    /predict API      │
                    │ Pydantic → Spark → ML│
                    └──────────────────────┘
```

## 🔄 Data Pipeline

The data pipeline follows a layered architecture:

### Bronze

Raw data is ingested with PySpark and stored without applying business-level transformations.

### Silver

The data is cleaned and validated before being passed to downstream processing.

### Gold

The Gold layer prepares the dataset for machine learning.

This includes:

* Selecting final loan outcomes
* Creating the binary `target`
* Removing identifiers
* Removing post-origination information that could cause data leakage
* Removing unsuitable columns
* Converting `emp_length` into a numerical representation
* Preparing the final machine-learning dataset

### Feature Engineering

Date-based features are generated during feature engineering, including:

* `issue_year`
* `issue_month`
* `credit_history_months`

The original date columns are then removed after the required features are derived.

## 🤖 Machine Learning

The current model uses Spark ML's **Logistic Regression** classifier.

The preprocessing pipeline includes:

* Numerical missing-value imputation
* Categorical indexing
* One-hot encoding
* Feature vector assembly
* Optional numerical scaling

Categorical features use:

```text
StringIndexer(handleInvalid="keep")
```

This allows previously unseen categorical values to be handled during inference.

The model is trained as part of a Spark ML `Pipeline`, keeping preprocessing and model inference together.

## 📊 Model Evaluation & Validation

Model validation is separated from model training.

The validation process evaluates model performance using classification metrics and applies configured promotion criteria.

The promotion logic considers:

* Recall
* Precision
* F1 score

A candidate model must satisfy the configured performance requirements before it can be registered and promoted.

## 🧪 MLflow

MLflow is used for:

* Experiment tracking
* Model artifact logging
* Model registration
* Model version management
* Champion model promotion

The registered model is:

```text
LoanDefaultModel
```

The production model is identified using the MLflow alias:

```text
champion
```

This allows the inference service to load the currently promoted model without hard-coding a specific model version.

For example:

```python
mlflow.spark.load_model(
    "models:/LoanDefaultModel@champion"
)
```

The current Champion model is **Version 5**.

## 🔐 Model Promotion

Model promotion follows a controlled process:

```text
Candidate Model
      │
      ▼
Validation
      │
      ├── Failed ──► Stop
      │
      ▼
Register Model
      │
      ▼
Compare with Champion
      │
      ├── Does not qualify ──► Keep Champion
      │
      ▼
Promote Candidate
      │
      ▼
Champion Alias Updated
```

This prevents every newly trained model from automatically becoming the production model.

## 🧪 Testing

Unit tests are implemented using `pytest`.

The CI-specific dependencies are kept separate from the full development environment:

```text
requirements-ci.txt
```

CI runs the automated tests whenever changes are pushed to GitHub.

## ⚙️ Continuous Integration

GitHub Actions is used to automate CI.

The workflow:

1. Checks out the repository
2. Sets up Python
3. Installs CI dependencies
4. Runs the automated tests

This ensures that changes are validated before being considered complete.

## 🐳 Docker

The inference service is containerized using Docker.

The Docker image contains:

* Python 3.11
* Java 21 runtime
* PySpark
* MLflow
* FastAPI
* Uvicorn

The Docker image is built from:

```text
docker/Dockerfile
```

Build the image with:

```powershell
docker build -t loan-default-api:latest -f docker\Dockerfile .
```

Run the API container with:

```powershell
docker run -d `
    --name loan-default-api `
    -p 8000:8000 `
    -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
    loan-default-api:latest
```

## 🚀 FastAPI Inference

The API exposes a health endpoint:

```text
GET /health
```

Example:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

Response:

```json
{
    "status": "healthy"
}
```

### Prediction Endpoint

```text
POST /predict
```

The API accepts loan features through a Pydantic request model.

The request schema provides:

* Numerical type validation
* Automatic conversion of compatible numeric strings
* Validation errors for invalid values
* Support for missing numerical values

For example:

```json
{
    "loan_amnt": 10000.0,
    "funded_amnt": 10000.0,
    "funded_amnt_inv": 10000.0,
    "term": 36.0,
    "int_rate": 10.78,
    "grade": "B",
    "sub_grade": "B4",
    "annual_inc": 54000.0,
    "issue_d": "2016-01-01",
    "earliest_cr_line": "2000-04-01"
}
```

The API then:

```text
JSON Request
     ↓
Pydantic Validation
     ↓
Spark DataFrame
     ↓
Date Feature Engineering
     ↓
MLflow Champion Model
     ↓
Prediction
```

The response contains:

```json
{
    "prediction": 0,
    "probability": 0.19207743392209742
}
```

## 📁 Project Structure

```text
CI_CD_END_TO_END/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── docker/
│   └── Dockerfile
│
├── docs/
│
├── notebooks/
│
├── scripts/
│
├── src/
│   ├── config/
│   ├── ingestion/
│   ├── ml/
│   ├── pipeline/
│   ├── split/
│   ├── transformation/
│   ├── utils/
│   └── inference/
│
├── tests/
│
├── requirements.txt
├── requirements-ci.txt
├── requirements-docker.txt
├── README.md
└── .gitignore
```

## 🛠️ Technologies

| Technology     | Purpose                                 |
| -------------- | --------------------------------------- |
| Python         | Application and ML development          |
| PySpark        | Distributed data processing and ML      |
| Delta Lake     | Layered data storage                    |
| MLflow         | Experiment tracking and model lifecycle |
| Scikit-learn   | ML utilities and evaluation             |
| XGBoost        | Machine learning experimentation        |
| FastAPI        | REST API                                |
| Pydantic       | API request validation                  |
| Docker         | Containerization                        |
| GitHub Actions | Continuous Integration                  |
| Git            | Version control                         |

## 🎯 Current Project Status

The core end-to-end implementation is complete.

### Completed

* [x] Environment setup
* [x] Data ingestion
* [x] Bronze processing
* [x] Silver processing and validation
* [x] Gold transformation
* [x] Feature engineering
* [x] Train/test splitting
* [x] Model training
* [x] Model evaluation
* [x] Model validation
* [x] MLflow experiment tracking
* [x] MLflow model registration
* [x] Champion model promotion
* [x] Automated tests
* [x] GitHub Actions CI
* [x] Dockerized inference service
* [x] FastAPI API
* [x] Pydantic request validation
* [x] MLflow Champion model loading
* [x] End-to-end Docker prediction

## ✅ End-to-End Verification

The final Dockerized inference path has been successfully tested:

```text
Gold Dataset
     ↓
JSON Request
     ↓
FastAPI
     ↓
Pydantic Validation
     ↓
Spark
     ↓
MLflow Champion Model
     ↓
Prediction
```

The container successfully returned a prediction:

```text
prediction: 0
probability: 0.19207743392209742
```

GitHub Actions CI is also passing successfully.

## 📌 Key MLOps Concepts Demonstrated

This project demonstrates practical MLOps concepts including:

* Modular ML pipeline design
* Separation of data, ML, and API responsibilities
* Data leakage prevention
* Feature engineering
* Model validation gates
* Model registry and versioning
* Champion model management
* Automated testing
* CI/CD
* Containerized inference
* API-level input validation
* Reproducible model deployment
* Separation between model training and inference

## 👤 Author

**Revanth Goli**

End-to-end MLOps / Machine Learning project demonstrating production-oriented ML engineering, model lifecycle management, CI/CD, and containerized inference.
