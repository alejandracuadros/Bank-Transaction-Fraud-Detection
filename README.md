# **Fraud Detection in Highly Imbalanced Datasets**

## **Introduction**

Fraud detection in credit card transactions is a crucial challenge in financial security. This project aims to develop a predictive model that accurately identifies fraudulent transactions in a dataset with a high class imbalance. The dataset includes 284,807 transactions, of which only 492 are fraudulent (0.172%).

## **Dataset**

- Transactions from European cardholders in September 2013.
- 30 anonymized features extracted using Principal Component Analysis (PCA).
- Additional features: `Time` (seconds elapsed since first transaction) and `Amount` (transaction value).
- Target variable: `Class` (0 = Non-Fraud, 1 = Fraud).

## **Project Workflow**

1. **Exploratory Data Analysis (EDA)**

   - Data distribution analysis
   - Feature correlation
   - Fraudulent transaction patterns

2. **Data Preprocessing**

   - Feature scaling (StandardScaler)
   - Feature engineering for improved model performance

3. **Handling Imbalance**

   - Under-sampling majority class
   - Over-sampling using **SMOTE (Synthetic Minority Over-sampling Technique)**

4. **Model Training & Evaluation**

   - **Logistic Regression**
   - **Random Forest Classifier**
   - **XGBoost Classifier**
   - **Isolation Forest** (Anomaly Detection)
   - **One-Class SVM** (Anomaly Detection)
   - Performance evaluation using:
     - Precision-Recall Curve
     - ROC-AUC Score
     - Confusion Matrix

5. **Model Explainability**

   - **SHAP (SHapley Additive Explanations)** for feature importance analysis
   - **LIME (Local Interpretable Model-agnostic Explanations)** for local instance predictions

## **Requirements**

Install required dependencies before running the project:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap lime
```

## **Usage**

1. Load the dataset:
   ```python
   import pandas as pd
   data = pd.read_csv('creditcard.csv')
   ```
2. Perform EDA and preprocessing.
3. Train the models and evaluate their performance.
4. Interpret model results using SHAP and LIME.

## **Results & Observations**

- **XGBoost** demonstrated the best performance in handling class imbalance.
- **SMOTE** significantly improved recall but slightly reduced precision.
- **SHAP & LIME** provided critical insights into fraud patterns and model decisions.

## **Conclusion**

This project highlights the effectiveness of various machine learning techniques for fraud detection. Handling class imbalance is critical to ensure robust model performance in real-world applications. Future work includes exploring deep learning approaches and real-time fraud detection systems.

## **Author**

**Alejandra Cuadros**

