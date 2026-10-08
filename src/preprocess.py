import pandas as pd
import numpy as np

import joblib
from sklearn.utils import resample

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def prepare_demand_data(df, sequence_length = 24):

    data = df.copy()

    data['order_timestamp'] = pd.to_datetime(data['order_timestamp'])

    hourly = (data.groupby(data['order_timestamp'].dt.floor('h')).agg(
            order_count = ('order_count','sum'),
            is_weekend = ('is_weekend','first'),
            weather = ('weather', lambda x: x.mode()[0]),
            traffic_level = ('traffic_level', lambda x: x.mode()[0]))
            .reset_index())

    hourly = hourly.sort_values('order_timestamp')

    hourly = hourly.set_index('order_timestamp')
    hourly = hourly.asfreq('h')
    hourly['order_count'] = hourly['order_count'].fillna(0)
    hourly['is_weekend'] = hourly['is_weekend'].ffill().bfill()
    hourly['weather'] = hourly['weather'].ffill().bfill()
    hourly['traffic_level'] = hourly['traffic_level'].ffill().bfill()
    hourly = hourly.reset_index()


    hourly['order_hour'] = hourly['order_timestamp'].dt.hour
    hourly['day_of_week'] = hourly['order_timestamp'].dt.dayofweek


    hourly = pd.get_dummies(hourly, columns = ['weather', 'traffic_level'],dtype = int)

    
    feature_columns = [col for col in hourly.columns if col != 'order_timestamp']

    X = hourly[feature_columns].values
    y = hourly['order_count'].values
    
    X_sequences = []
    y_sequences = []

    for i in range(sequence_length, len(X)):

        X_sequences.append(X[i-sequence_length:i])

        y_sequences.append(y[i])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)


    split_index = int(len(X_sequences) * 0.7)

    X_train = X_sequences[:split_index]
    X_test = X_sequences[split_index:]

    y_train = y_sequences[:split_index]
    y_test = y_sequences[split_index:]

   
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1,X_train.shape[2])
    X_test_2d = X_test.reshape(-1,X_test.shape[2])

    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform( X_test_2d)

    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    joblib.dump(scaler,'model/demand_scaler.pkl')

    return (X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)

def prepare_delivery_data(df):
    data = df[df['actual_delivery_time_minutes'].notna()].copy()

    delivery_df = data[['distance_km','traffic_level','restaurant_preparation_time_minutes','estimated_delivery_time_minutes','delivery_partner_experience_months',
                             'delivery_partner_rating','actual_delivery_time_minutes','weather','is_weekend']]
    
    delivery_df = pd.get_dummies(delivery_df, columns = ['traffic_level','weather'],dtype = int)


    X = delivery_df.drop(columns = ['actual_delivery_time_minutes'])
    y = delivery_df['actual_delivery_time_minutes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    joblib.dump(scaler, 'model/delivery_time_scaler.pkl')

    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled, y_train.values, y_test.values, X.columns.tolist())


def prepare_sentiment_data(df):

    data = df[df['review'].notna()].copy()

    review_df = data[['review', 'review_sentiment']]

    review_unique_df = review_df.drop_duplicates(subset = 'review').copy()

    X = review_unique_df['review'].astype(str).to_numpy()

    y = review_unique_df['review_sentiment'].astype(str).to_numpy()

    label_map = {'Negative': 0,'Neutral': 1,'Positive': 2}

    y = np.array([label_map[label]for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    y_train_original = y_train.copy()

    train_df = pd.DataFrame({'review': X_train,'review_sentiment': y_train})

    class_0 = train_df[train_df['review_sentiment'] == 0]
    class_1 = train_df[train_df['review_sentiment'] == 1]
    class_2 = train_df[train_df['review_sentiment'] == 2]

    max_count = max(len(class_0),len(class_1),len(class_2))

    class_0_up = resample(class_0,replace = True,n_samples = max_count,random_state = 42)
    class_1_up = resample(class_1,replace = True,n_samples = max_count,random_state = 42)
    class_2_up = resample(class_2,replace = True,n_samples = max_count,random_state = 42)

    train_balanced = pd.concat([class_0_up, class_1_up, class_2_up])

    X_train = train_balanced['review'].to_numpy()
    y_train = train_balanced['review_sentiment'].to_numpy()

    print("Balanced training distribution:")
    print(np.bincount(y_train))

    return (X_train, X_test, y_train, y_test, label_map, y_train_original)