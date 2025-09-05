![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-green)
![Pandas](https://img.shields.io/badge/Pandas-1.5-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6-orange)
![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn-0.10-lightgrey)

# Credit Card Fraud Detection using Machine Learning

## Project Overview
This project implements a machine learning system to detect fraudulent credit card transactions using **Random Forest classifier**. The system handles severe class imbalance (0.17% fraud cases) through undersampling techniques, achieving high accuracy in identifying fraudulent activities.

## Tasks Performed
- **Data Preprocessing:** Used `RandomUnderSampler` from imbalanced-learn to handle class imbalance
- **Model Training:** Trained a `RandomForestClassifier` on the balanced training data
- **Model Evaluation:** 
  - Computed **Accuracy, Precision, Recall, and F1-Score** on the test set
  - Generated **Confusion Matrix** and **Classification Report**
  - Visualized **Feature Importance** for fraud detection
- **Data Visualization:** Created charts showing data distribution and model performance

## Key Insights
- Random Forest effectively handles extreme class imbalance in financial data
- The model achieves high recall rate, ensuring most fraudulent transactions are detected
- Feature importance analysis reveals key factors contributing to fraudulent activities
- Undersampling technique significantly improves model performance on minority class

## 📊 Results
| Metric | Score |
|--------|-------|
| **Accuracy** | 98.05% |
| **Precision** | 0.073 |
| **Recall** | 0.878 |
| **F1-Score** | 0.135 |

## Files in this Repository
- `Credit_Card_Fraud_Detection_Project.ipynb` – Complete Jupyter Notebook with code, analysis, and visualizations
- `README.md` – Project documentation (this file)
- `Screenshots1/` – Folder containing output screenshots:
  - `Cell2_Coding.png` – Data visualization code
  - `Cell2_Output.png` – Data distribution charts
  - `Cell5_Coding.png` – Model evaluation code
  - `Cell5_Output.png` – Performance metrics and feature importance

## How to Run
1. Open `Credit_Card_Fraud_Detection_Project.ipynb` in **Google Colab** or Jupyter Notebook
2. Run all cells sequentially to see:
   - Dataset loading and exploration
   - Data preprocessing with undersampling
   - Model training with Random Forest
   - Prediction and evaluation metrics
   - Visualization outputs

## Visuals
### Data Analysis Code
![Data Analysis Code](Screenshots1/Cell_2_Coding.png)

### Data Distribution Charts  
![Data Distribution](Screenshots1/Cell_2_Output.png)

### Model Evaluation Code
![Model Evaluation Code](Screenshots1/Cell_5_Coding.png)

### Performance Metrics & Feature Importance
![Model Performance](Screenshots1/Cell_5_Output.png)

## Dependencies
- **Python 3.x**
- **Pandas & NumPy** – Data manipulation and analysis
- **Scikit-learn** – Machine learning models and evaluation
- **Imbalanced-learn** – Handling class imbalance
- **Matplotlib & Seaborn** – Data visualization
- **Google Colab** – Development environment

## Dataset Information
- **Source:** Kaggle Credit Card Fraud Detection Dataset
- **Samples:** 284,807 transactions
- **Features:** 30 numerical features (V1-V28, Time, Amount)
- **Target Variable:** Class (0 = Legitimate, 1 = Fraudulent)
- **Fraud Rate:** 0.17% (492 fraudulent transactions)

## Next Steps & Learning Tips
- **Try Other Models:** Experiment with XGBoost, Logistic Regression, or Neural Networks for comparison
- **Advanced Sampling:** Test SMOTE or ADASYN instead of RandomUnderSampler
- **Feature Engineering:** Create new features or perform deeper correlation analysis
- **Hyperparameter Tuning:** Use GridSearchCV to optimize Random Forest parameters
- **Real-time Detection:** Explore streaming data implementation for real-time fraud detection
- **Model Deployment:** Learn to deploy the model as a web service using Flask or FastAPI

## Acknowledgments
- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Built as part of a machine learning project to demonstrate fraud detection techniques
