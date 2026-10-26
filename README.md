# AI-Powered Food Delivery Intelligence Platform

An end-to-end Deep Learning project for food demand prediction, delivery time prediction, and customer review sentiment analysis.

## Live Demo

* **Frontend:** [Streamlit App](https://food-delivery-frontend-w40j.onrender.com/)
* **Backend:** [FastAPI](https://food-delivery-backend-ijvg.onrender.com)
* **API Documentation:** [Swagger UI](https://food-delivery-backend-ijvg.onrender.com/docs)

## Project Overview

This project uses Deep Learning to solve three food delivery problems:

1. **Food Demand Prediction** – Uses LSTM to forecast food demand.
2. **Delivery Time Prediction** – Uses an Artificial Neural Network (ANN) to estimate delivery time.
3. **Review Sentiment Analysis** – Uses FastText embeddings and LSTM to classify customer reviews as Positive, Neutral, or Negative.

## Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* TensorFlow, Keras
* FastText
* FastAPI
* Streamlit
* MLflow
* Docker, Docker Compose
* Render

## Project Workflow

Data Collection → Data Preprocessing → EDA → Model Training → Model Evaluation → API Development → Streamlit Frontend → Docker → Render Deployment

## Key Features

* Food demand forecasting
* Delivery time prediction
* Customer review sentiment analysis
* Exploratory Data Analysis (EDA)
* Model configuration comparison
* MLflow experiment tracking
* FastAPI backend
* Interactive Streamlit frontend
* Docker-based deployment

## API Endpoints

| Method | Endpoint             | Description              |
| ------ | -------------------- | ------------------------ |
| GET    | `/`                  | Backend health check     |
| POST   | `/predict/demand`    | Predict food demand      |
| POST   | `/predict/delivery`  | Predict delivery time    |
| POST   | `/predict/sentiment` | Analyze review sentiment |

## Run Locally

Clone the repository:

```bash
git clone https://github.com/n-a-nd-a-na/AI-Powered-Food-Delivery-Intelligence-Platform.git
```

Navigate to the project folder:

```bash
cd AI-Powered-Food-Delivery-Intelligence-Platform
```

Run using Docker Compose:

```bash
docker compose up --build
```

## Deployment

The frontend and backend are deployed as separate Docker-based services on Render.

* Frontend: Streamlit
* Backend: FastAPI

## Future Improvements

* Improve prediction accuracy.
* Add model monitoring.
* Automate model retraining.
* Improve API validation and error handling.

## Author

**Nandana Jayan**

BSc Mathematics | AI & Data Science

[GitHub Profile](https://github.com/n-a-nd-a-na)
