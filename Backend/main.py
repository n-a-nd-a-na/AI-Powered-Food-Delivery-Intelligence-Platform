import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

from Backend.schemas import DemandInput, DeliveryInput, SentimentInput


app = FastAPI(title = 'Food Delivery Intelligence API',
            description = 'API for Demand, Delivery Time and Review Sentiment Prediction')


df = pd.read_csv('data/food_delivery_processed.csv')
df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])

demand_scaler = joblib.load('model/demand_scaler.pkl')
demand_model = load_model('model/model_demand.keras')

delivery_scaler = joblib.load('model/delivery_time_scaler.pkl')
delivery_model = load_model('model/model_time.keras')

sentiment_tokenizer = joblib.load('model/sentiment_fasttext_tokenizer.pkl')
sentiment_model = load_model('model/model_sentiment.keras')

demand_features = ['order_count','is_weekend','order_hour','day_of_week','weather_Clear','weather_Cloudy','weather_Rain','weather_Storm',
                    'traffic_level_High','traffic_level_Low','traffic_level_Medium']


delivery_features = ['distance_km','restaurant_preparation_time_minutes','estimated_delivery_time_minutes','delivery_partner_experience_months',
                        'delivery_partner_rating','is_weekend','traffic_level_High','traffic_level_Low','traffic_level_Medium',
                        'weather_Clear','weather_Cloudy','weather_Rain','weather_Storm']

@app.get('/')
def home():

    return {'message': 'Food Delivery Intelligence API is running',
            'tasks': ['Demand Prediction','Delivery Time Prediction','Review Sentiment Analysis']}


@app.post('/predict/demand')
def predict_demand(data: DemandInput):

    timestamp = data.order_timestamp

    previous_24 = df[(df['order_timestamp'] < timestamp) & (df['order_timestamp'] >= timestamp - pd.Timedelta(hours = 24))].copy()

    previous_24['hour'] = previous_24['order_timestamp'].dt.floor('h')

    hourly_previous = (previous_24.groupby('hour').agg(
            order_count = ('order_count', 'sum'),
            is_weekend = ('is_weekend', 'first'),
            weather = ('weather', lambda x: x.mode()[0]),
            traffic_level = ('traffic_level', lambda x: x.mode()[0])).reset_index())

    hourly_previous['order_hour'] = hourly_previous['hour'].dt.hour
    hourly_previous['day_of_week'] = hourly_previous['hour'].dt.dayofweek

    hourly_previous = pd.get_dummies(hourly_previous,columns = ['weather', 'traffic_level'],dtype = int )

    for feature in demand_features:
        if feature not in hourly_previous.columns:
            hourly_previous[feature] = 0

    sequence = hourly_previous[demand_features].values

    if len(sequence) != 24:
        return {'error': f'Only {len(sequence)} previous hourly records found. 24 are required.'}

    sequence_scaled = demand_scaler.transform(sequence)

    sequence_scaled = sequence_scaled.reshape(1, 24, len(demand_features))

    prediction = demand_model.predict(sequence_scaled)

    predicted_orders = float(prediction[0][0])

    return {'task': 'Demand Prediction','predicted_order_count': round(predicted_orders)}


@app.post('/predict/delivery') 
def predict_delivery(data: DeliveryInput): 
    row = {'distance_km': data.distance_km,
            'restaurant_preparation_time_minutes':data.restaurant_preparation_time_minutes,
            'estimated_delivery_time_minutes':data.estimated_delivery_time_minutes,
            'delivery_partner_experience_months':data.delivery_partner_experience_months, 
            'delivery_partner_rating':data.delivery_partner_rating, 'is_weekend': data.is_weekend, 
            'traffic_level_High': 0, 'traffic_level_Low': 0, 'traffic_level_Medium': 0, 'weather_Clear': 0, 'weather_Cloudy': 0, 'weather_Rain': 0, 'weather_Storm': 0}
    
    traffic_column = f'traffic_level_{data.traffic_level}'
    if traffic_column in row:
        row[traffic_column] = 1 

    weather_column = f'weather_{data.weather}'
    if weather_column in row:
         row[weather_column] = 1 
         
    X = pd.DataFrame([row],columns = delivery_features) 

    X_scaled = delivery_scaler.transform(X) 

    prediction = delivery_model.predict(X_scaled)

    predicted_time = float(prediction[0][0])

    return {'task': 'Delivery Time Prediction', 'predicted_delivery_time_minutes': round(predicted_time, 2)}

@app.post('/predict/sentiment')
def predict_sentiment(data: SentimentInput):

    sequence = sentiment_tokenizer.texts_to_sequences([data.review])

    padded = pad_sequences(sequence,maxlen = 100, padding = 'post')

    prediction = sentiment_model.predict(padded)

    predicted_class = int(np.argmax(prediction, axis = 1)[0])

    labels = {0: 'Negative',1: 'Neutral',2: 'Positive'}

    confidence = float(np.max(prediction))

    return {'task': 'Review Sentiment Analysis',
        'sentiment': labels[predicted_class],
        'confidence': round(confidence, 4)}