# Patient-Risk-Prediction
Machine Learning-based heart disease prediction with the help of different features using K-Nearest Neighbors (KNN) with Hyperparameter Tuning, Model Evaluation, and Streamlit Deployment.
## Features

- Data Cleaning
- Feature Scaling
- KNN Classification
- Hyperparameter Tuning
- Confusion Matrix
- Streamlit Deployment

## Project Workflow

## Model Performance

Accuracy: 83%

## Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

## Liberaries

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit

## Hyperparameter Tuning

GridSearchCV was used to determine the best K value.

Best Parameters:

K = 7

Metric = Euclidean

Weights = Uniform

## Run Project

pip install -r requirements.txt

streamlit run app/streamlit_app.py
