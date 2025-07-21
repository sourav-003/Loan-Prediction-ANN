# 🏦 Loan Prediction System using Deep Learning

A robust deep learning project built to automate the process of loan approval predictions using structured data from banks. This project uses both **Functional API** and **Sequential API** models in TensorFlow/Keras and includes complete data preprocessing, visualization, evaluation metrics, and enhancement strategies.

---

## 📌 Table of Contents

- [📖 Overview](#-overview)
- [📁 Dataset Description](#-dataset-description)
- [🧼 Data Preprocessing](#-data-preprocessing)
- [🧠 Model Architectures](#-model-architectures)
  - [Functional API](#functional-api-model)
  - [Sequential API](#sequential-api-model)
- [📊 Evaluation Metrics](#-evaluation-metrics)
- [📈 Visualizations](#-visualizations)
- [🚀 Enhancements & Ideas](#-enhancements--ideas)
- [🛠 Installation & Setup](#-installation--setup)
- [🧪 How to Run](#-how-to-run)
- [📂 Folder Structure](#-folder-structure)
- [🔒 License](#-license)

---

## 📖 Overview

The goal of this project is to predict whether a loan will be approved (`Loan_Status`) using applicant data. It is a common use case in finance and banking where automation can significantly improve efficiency, reduce human bias, and scale decision-making.

This project is ideal for:
- Practicing end-to-end machine learning pipeline development
- Deep learning applications in structured tabular data
- Exploratory Data Analysis (EDA) and Feature Engineering
- Comparing Functional vs Sequential Keras models

---

## 📁 Dataset Description

- **Dataset Name**: `loan_data.csv`
- **Source**: Kaggle / Public Domain
- **Total Rows**: 614
- **Target Column**: `Loan_Status`

### Feature Summary

| Feature             | Type        | Description |
|---------------------|-------------|-------------|
| Gender              | Categorical | Male/Female |
| Married             | Categorical | Marital Status |
| Dependents          | Categorical | No. of dependents |
| Education           | Categorical | Graduate/Not Graduate |
| Self_Employed       | Categorical | Employment status |
| ApplicantIncome     | Numerical   | Income of applicant |
| CoapplicantIncome   | Numerical   | Income of co-applicant |
| LoanAmount          | Numerical   | Loan Amount (₹) |
| Loan_Amount_Term    | Numerical   | Repayment term (months) |
| Credit_History      | Numerical   | Good or bad history |
| Property_Area       | Categorical | Urban/Semiurban/Rural |
| Loan_Status         | Categorical | Target variable |

---

## 🧼 Data Preprocessing

### ✅ Steps Followed:
- Removed identifier column: `Loan_ID`
- Imputed missing values:
  - Categorical → Mode
  - Numerical → Median/Mean
- Label encoded binary categorical values
- One-hot encoded nominal categorical values
- Scaled numerical columns (if needed)
- Train-Test split (80-20)
- Converted labels for binary classification

---

## 🧠 Model Architectures

### Functional API Model

The Functional API was used to:
- Handle different input feature types (numerical vs categorical)
- Build modular, flexible networks
- Include dropout layers for regularization

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

input_layer = Input(shape=(input_dim,))
x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
```

## 🧠 Sequential API Model

A simplified but powerful deep learning model using TensorFlow's Sequential API.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```
---

## 📊 Evaluation Metrics
Used comprehensive metrics to evaluate performance:

- ✅ **Accuracy**
- 🔁 **Precision & Recall**
- 🧮 **F1-Score**
- 📉 **Confusion Matrix**

> ⚠️ In loan approval tasks, **Recall** is prioritized to reduce false negatives — i.e., wrongly rejecting eligible applicants.

---

## 📈 Visualizations

Visual analysis includes:

- ✅ **Training vs Validation Accuracy/Loss Curves**
- 📉 **Confusion Matrix (Seaborn Heatmap)**
- 📈 **Distribution of Numerical and Categorical Features**

---

## 🚀 Enhancements & Ideas

### ✅ Feature Engineering
- Income-to-loan ratio
- EMI per dependent

### ✅ Hyperparameter Tuning
- Learning rate adjustments
- Early stopping
- Regularization techniques (Dropout, L2)

### ✅ Handling Class Imbalance
- SMOTE (Synthetic Minority Oversampling Technique)
- Weighted loss functions

### ✅ Deployment
- Streamlit or Flask web app

### ✅ Model Explainability
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-Agnostic Explanations)

---

## 🛠 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/loan-prediction-dl.git
cd loan-prediction-dl
```

# (Recommended) Create virtual environment
conda create -n loanenv python=3.9
conda activate loanenv

---

# Install requirements
pip install -r requirements.txt

---

## 📂 Folder Structure

```bash
loan-prediction-dl/
│
├── data/
│   └── loan_data.csv
│
├── Loan_Prediction.ipynb
├── requirements.txt
├── README.md
└── saved_model/
    └── model.h5 (optional)
```
---

## 🔒 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute the code with proper attribution.

---

## 👨‍💻 Author

**Sourav Kumar**  

- [LinkedIn](https://linkedin.com/in/sourav-kumar-5814341b8)
- [GitHub](https://github.com/yourusername)
