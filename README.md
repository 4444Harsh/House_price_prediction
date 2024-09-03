# House Prices Prediction

This repository contains a Jupyter notebook that performs house price prediction using various machine learning models and selects the best model based on the R² score.

## Project Overview

The goal of this project is to build and evaluate several machine learning models to predict house prices using a dataset of house features. The best-performing model is selected based on the R² score and used to generate predictions for the test dataset.

## Dataset

The project uses the following datasets:
- **train.csv**: The training dataset contains features of houses along with the target variable `SalePrice`.
- **test.csv**: The test dataset contains features of houses without the target variable `SalePrice`. The goal is to predict `SalePrice` for this dataset.
- **sample_submission.csv**: This file provides the format for the submission file containing `Id` and the predicted `SalePrice`.

## Models

The following models are evaluated in the project:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

## Workflow

1. **Data Preprocessing**:
   - Handle missing values using imputation strategies.
   - Encode categorical variables using one-hot encoding.
   - Scale numerical features to normalize them.

2. **Model Training and Evaluation**:
   - Multiple models are trained using the training dataset.
   - The performance of each model is evaluated using the R² score on a validation dataset.
   - The model with the highest R² score is selected as the best model.

3. **Saving the Best Model**:
   - The best-performing model is saved using `joblib` for future use.

4. **Generating Predictions**:
   - The best model is used to predict house prices on the test dataset.
   - The predictions are written to a `submission.csv` file in the required format.

## Files

- **train.csv**: The training dataset.
- **test.csv**: The test dataset.
- **sample_submission.csv**: The sample submission format.
- **best_model.pkl**: The best-performing model saved for future use.
- **submission.csv**: The file containing predictions for the test dataset.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/house-prices-prediction.git
    cd house-prices-prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook house_prices_prediction.ipynb
    ```

4. Once the notebook runs successfully, it will:
   - Train and evaluate multiple models.
   - Save the best model as `best_model.pkl`.
   - Generate a `submission.csv` file for submission.

## Evaluation Metric

The project uses the R² score as the evaluation metric to measure the performance of the models. The R² score indicates how well the model predicts the variance in the target variable.

## Requirements

- Python 3.x
- Jupyter Notebook
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

Install dependencies using:
```bash
pip install pandas scikit-learn matplotlib seaborn joblib
