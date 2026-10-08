# AI-Powered Food Delivery Intelligence Platform

An end-to-end **Deep Learning-based Food Delivery Intelligence Platform** that provides predictions and insights for food delivery operations.

The project combines three different deep learning tasks into a single application:

1. **Food Demand Prediction** using LSTM
2. **Delivery Time Prediction** using ANN
3. **Review Sentiment Analysis** using FastText + LSTM

The complete application is integrated with a **FastAPI backend** and a **frontend interface**, and the application is containerized using **Docker and Docker Compose**.

---

# Project Objectives

The main objectives of this project are:

* Predict future food delivery demand.
* Predict delivery time for an order.
* Analyze customer review sentiment.
* Perform task-wise Exploratory Data Analysis (EDA).
* Train and compare different deep learning model configurations.
* Track experiments and model performance using MLflow.
* Integrate all three prediction tasks into a single application.
* Deploy the application using Docker and Docker Compose.

---

# Project Tasks

## 1.Food Demand Prediction

### Model

**LSTM (Long Short-Term Memory)**

LSTM is used for predicting future food delivery demand based on historical order patterns.

### Input Features

The demand prediction task uses features such as:

* Order timestamp
* Weekend information
* Weather
* Traffic related information
* Order count

### Target

```text
Order Count
```

Historical order information is used to create sequential data for the LSTM model.

### Why LSTM?

LSTM is suitable for sequential data because it can learn patterns from previous time steps and use them to predict future values.

### Prediction Output

The model predicts the expected number of food orders.

Example:

```text
Predicted Orders: 1650
```

---

# 2.Delivery Time Prediction

### Model

**Artificial Neural Network (ANN)**

The ANN model is used to predict the delivery time of a food order.

### Input Features

The model uses relevant order, restaurant, delivery partner, distance, weather, and traffic related features.

Examples include:

* Distance
* Weather
* Traffic level
* Restaurant preparation time
* Delivery partner experience
* Delivery partner rating
* Restaurant rating
* Items count
* Order related information

### Target

```text
Actual Delivery Time
```

### Prediction Output

The model predicts the delivery duration.

Example:

```text
Predicted Delivery Time: 30 minutes
```

---

# 3.Review Sentiment Analysis

### Model

**FastText + LSTM**

FastText is used for generating word representations, and the LSTM model is used to learn patterns from the review text for sentiment classification.

### Input

```text
Customer Review
```

### Model Flow

```text
Customer Review
       ↓
FastText
       ↓
LSTM
       ↓
Sentiment Prediction
       ↓
Positive / Negative
```

### Output

The model classifies the review into:

```text
Positive
Negative
Neutral
```

### Example

```text
Review:
"The food was really tasty"

Predicted Sentiment:
Positive
confidence : 99.6%
```

---

# Exploratory Data Analysis

Exploratory Data Analysis was performed before model development to understand the dataset and identify useful patterns.

## General EDA

The following analysis was performed:

* Dataset information
* Statistical summary
* Missing value analysis
* Duplicate checking
* Categorical feature analysis
* Numerical feature analysis
* Outlier detection
* Correlation analysis

---

# Task-wise EDA

EDA was performed separately for each project task.

## Demand Prediction EDA

The demand-related analysis includes:

* Daily order trends
* Order count distribution
* Weekend vs weekday demand
* Weather and order demand
* Traffic and order demand
* Time based demand patterns

---

## Delivery Time Prediction EDA

The delivery-time analysis includes:

* Distance vs delivery time
* Traffic vs delivery time
* Weather vs delivery time
* Restaurant preparation time
* Delivery partner experience
* Delivery partner rating

---

## Review Sentiment Analysis EDA

The review-related analysis includes:

* Review distribution
* Sentiment distribution
* Review related patterns

# Technologies Used

## Programming

* Python

## Data Analysis

* Pandas
* NumPy
* Matplotlib
* Seaborn

## Deep Learning

* TensorFlow
* Keras
* LSTM
* Artificial Neural Network

## Text Representation / Sentiment

* FastText
* LSTM

## Machine Learning

* Scikit-learn

## Experiment Tracking

* MLflow

## Backend

* FastAPI
* Uvicorn

## Frontend

* Streamlit

## Deployment

* Docker
* Docker Compose

---


# Docker & Docker Compose

The application is containerized using **Docker**.

Docker allows the application and its dependencies to run consistently across different environments.

Docker Compose is used to manage the application services together.

---

## Running the Project with Docker

Make sure **Docker Desktop** is installed and running.

Build and start the application:

```bash
docker compose up --build
```

# Application Access

After the Docker containers are running, the application can be accessed through the configured ports.

### Backend

```text
http://localhost:8000
```

### FastAPI Documentation

```text
http://localhost:8000/docs
```

### Frontend

```text
http://localhost:8501
```

> The actual ports depend on the configuration in `docker-compose.yml`.

---

# FastAPI Backend

The backend is developed using **FastAPI**.

The backend acts as the communication layer between the frontend and the trained prediction models.

The API handles prediction requests for:

* Demand Prediction
* Delivery Time Prediction
* Review Sentiment Analysis

FastAPI provides interactive API documentation through:

```text
http://localhost:8000/docs
```

---

# Frontend

The project provides a unified frontend interface for the three prediction tasks.

The user can select the required task and provide the necessary input.

The frontend communicates with the FastAPI backend to obtain prediction results.

---

# Model Evaluation

Different configurations were experimented with for the three tasks.

## Demand Prediction

Evaluation metrics:

* MAE
* MSE
* RMSE
* R² Score

## Delivery Time Prediction

Evaluation metrics:

* MAE
* MSE
* RMSE
* R² Score

## Sentiment Analysis

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score

MLflow was used to track and compare the experiments.

---

# Project Structure

Food-Delivery/
│
├── backend/
│
├── frontend/
│
├── data/
│
├── models/
│
├── src/
│
├── notebooks/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md

---

# Future Improvements

Possible future improvements include:

* Real-time demand forecasting
* More advanced time-series models
* Transformer-based sentiment models
* Real-time traffic integration
* Real-time weather integration
* Restaurant-level demand forecasting
* Driver-level optimization
* Cloud deployment
* Model monitoring
* Automated model retraining
* Advanced analytics dashboard

---

# Author

## Nandana

**Data Science & Artificial Intelligence**

This project demonstrates an end-to-end application combining:

**Data Analysis → Deep Learning → FastText + LSTM → MLflow → FastAPI → Frontend → Docker**

---

# Project Highlights

* LSTM-based demand prediction
* ANN-based delivery time prediction
* FastText + LSTM sentiment analysis
* Task-wise EDA
* MLflow experiment tracking
* FastAPI backend
* Unified frontend
* Docker containerization
* Docker Compose integration

---