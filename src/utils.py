import numpy as np
import mlflow

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,accuracy_score, precision_score, recall_score, f1_score)


def evaluate_regression(y_test, y_pred):
    
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae,"RMSE": rmse,"R2": r2}

    return metrics


def evaluate_classification(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, average='weighted', zero_division = 0)

    recall = recall_score(y_test, y_pred, average='weighted', zero_division = 0)

    f1 = f1_score(y_test, y_pred, average='weighted', zero_division = 0 )

    metrics = {"Accuracy": accuracy,"Precision": precision,"Recall": recall,"F1_Score": f1}

    return metrics


def log_parameters(parameters):
    for name, value in parameters.items():
        mlflow.log_param(name, value)

def log_model(model, model_name):
    mlflow.tensorflow.log_model(model,name = model_name)

def log_metrics(metrics):
    for name, value in metrics.items():
        mlflow.log_metric(name,float(value))


def print_metrics(metrics):
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")