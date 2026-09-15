# House Price Prediction

A machine learning project that predicts house sale prices using the Ames Housing dataset.

## Project Overview

This project demonstrates an end-to-end machine learning workflow for a regression problem, including data exploration, preprocessing, model training, evaluation, and model analysis.

## Dataset

Dataset: House Prices - Advanced Regression Techniques

The dataset was obtained from the Kaggle competition:
House Prices - Advanced Regression Techniques.

The dataset is not included in this repository. It can be downloaded from Kaggle.

The dataset contains information about residential properties, including features related to:

- Overall quality
- Living area
- Basement
- Garage
- Neighborhood
- Number of rooms
- Year built
- And many other property characteristics

Target variable:

`SalePrice`

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Joblib

## Workflow

### 1. Exploratory Data Analysis

- Examined dataset structure and statistics
- Analyzed the distribution of `SalePrice`
- Checked target skewness
- Studied feature correlations
- Visualized relationships between important features and house prices

### 2. Data Cleaning

- Analyzed missing values
- Distinguished between actual missing data and features that were absent
- Used appropriate imputation strategies

### 3. Preprocessing

- Separated numerical and categorical features
- Median imputation for numerical features
- Standardization of numerical features
- One-hot encoding of categorical features
- Used `ColumnTransformer` and `Pipeline`

### 4. Model Training

The following regression models were evaluated:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### 5. Hyperparameter Tuning

`RandomizedSearchCV` with 5-fold cross-validation was used to search for better Random Forest hyperparameters.

### 6. Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- R² Score

### Results

| Model | MAE | R² |
|---|---:|---:|
| Linear Regression | ~21,126 | ~0.443 |
| Random Forest | ~17,512 | ~0.891 |
| Tuned Random Forest | ~17,612 | ~0.889 |
| Gradient Boosting | ~17,123 | ~0.903 |

Gradient Boosting achieved the best performance on the held-out test set.

**Final Model:** Gradient Boosting Regressor

**R²:** ~0.903

**MAE:** ~17,123

## Feature Importance

Feature importance analysis showed that features such as:

- OverallQual
- GrLivArea
- GarageCars
- BsmtFinSF1
- TotalBsmtSF
- 1stFlrSF

were among the most influential features for the model.

## Project Structure

```text
House-Price-Prediction/
│
├── House_price_prediction.ipynb
├── app.py
├── sample_input.csv
├── house_price_model.pkl
├── house_price_preprocessor.pkl
├── requirements.txt
└── README.md

## How to Run

### Run the Jupyter Notebook

1. Clone the repository.
2. Download the dataset from Kaggle.
3. Place `train.csv` in the project directory.
4. Install the required Python libraries.
5. Open `House_price_prediction.ipynb` in Jupyter Notebook.
6. Run the notebook cells sequentially.

### Run the Web App

1. Install the required Python libraries.
2. Run the following command:

```bash
streamlit run app.py