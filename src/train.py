import numpy as np
import mlflow
import tensorflow as tf
import joblib

from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, GRU # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

from preprocess import (load_data, prepare_demand_data, prepare_delivery_data, prepare_sentiment_data)

from utils import (evaluate_regression, evaluate_classification, log_parameters, log_model, log_metrics,print_metrics)

mlflow.set_experiment('Food_Delivery_Intelligence_Experiment')

data_path = 'data/food_delivery_processed.csv'

df = load_data(data_path)

print('Dataset shape:', df.shape)

print('TASK 1 - DEMAND PREDICTION')

X_train, X_test, y_train, y_test, demand_features = prepare_demand_data(df, sequence_length = 24)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

with mlflow.start_run(run_name = 'Demand_Baseline'):
    baseline_prediction = np.full(len(y_test),y_train.mean())
    baseline_metrics = evaluate_regression(y_test,baseline_prediction)
    log_parameters({'task' : 'Demand Prediction', 'model' : 'Mean Baseline'})
    log_metrics(baseline_metrics)
    print('\nDemand Baseline:')
    print_metrics(baseline_metrics)


demand_models = [{'name': 'LSTM_128_D01_E50','units': 128,'dropout': 0.1,'epochs': 50,'batch_size': 16},
                {'name': 'LSTM_128_D02_E50','units': 128,'dropout': 0.2,'epochs': 50,'batch_size': 16},
                {'name': 'LSTM_128_D01_E75','units': 128,'dropout': 0.1,'epochs': 75,'batch_size': 16},
                {'name': 'LSTM_128_D02_E75','units': 128,'dropout': 0.2,'epochs': 75,'batch_size': 16},
                {'name': 'LSTM_128_D015_E100','units': 128,'dropout': 0.15,'epochs': 100,'batch_size': 16}]


for i in demand_models:

    print('\nTraining:', i['name'])

    with mlflow.start_run(run_name = i['name']):

        model = Sequential([LSTM(i['units'],input_shape = (X_train.shape[1], X_train.shape[2])),
                Dropout(i['dropout']),
                Dense(1)])

        model.compile(optimizer = 'adam',loss = 'mse')

        model.fit(X_train, y_train,epochs = i['epochs'],batch_size = i['batch_size'], validation_split = 0.1)

        y_pred = model.predict(X_test)

        metrics = evaluate_regression(y_test, y_pred)

        parameters = {'task': 'Demand Prediction','model': 'LSTM','units': i['units'],'dropout': i['dropout'],'epochs': i['epochs'],'batch_size': i['batch_size'],'sequence_length': 24}

        log_parameters(parameters)
        log_metrics(metrics)
        log_model(model,i['name'])
        print_metrics(metrics)

print('TASK 2 - DELIVERY TIME PREDICTION')

X_train, X_test, y_train, y_test, delivery_features = prepare_delivery_data(df)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

with mlflow.start_run(run_name = 'Delivery_Baseline'):
    baseline_prediction = np.full(len(y_test),y_train.mean())
    baseline_metrics = evaluate_regression(y_test,baseline_prediction)
    log_parameters({'task' : 'Delivery Time Prediction', 'model' : 'Mean Baseline'})
    log_metrics(baseline_metrics)
    print('\nDelivery Time Baseline:')
    print_metrics(baseline_metrics)

delivery_models = [{'name': 'ANN_32','units': 32,'dropout': 0.2,'epochs': 20,'batch_size': 32},
                    {'name': 'ANN_64','units': 64,'dropout': 0.2,'epochs': 30,'batch_size': 32},
                    {'name': 'ANN_64_Dropout','units': 64,'dropout': 0.3,'epochs': 30,'batch_size': 64},
                    {'name': 'ANN_128','units': 128,'dropout': 0.2,'epochs': 30,'batch_size': 32},
                    {'name': 'ANN_128_Dropout','units': 128,'dropout': 0.3,'epochs': 40,'batch_size': 32}]

for i in delivery_models:

    print('\nTraining:', i['name'])

    with mlflow.start_run(run_name = i['name']):

        model = Sequential([Dense(i['units'],activation = 'relu',input_shape = (X_train.shape[1],)),
        Dropout(i['dropout']),
        Dense(16,activation = 'relu'),
        Dense(1)])

        model.compile(optimizer = 'adam',loss = 'mse')

        model.fit(X_train, y_train, epochs = i['epochs'], batch_size = i['batch_size'], validation_split = 0.1)

        y_pred = model.predict(X_test)

        metrics = evaluate_regression(y_test, y_pred)

        parameters = {'task': 'Delivery Time Prediction','model': 'ANN','units': i["units"],'dropout': i['dropout'],'epochs': i['epochs'],'batch_size': i['batch_size']}

        log_parameters(parameters)
        log_metrics(metrics)
        log_model(model,i['name'])
        print_metrics(metrics)


print("TASK 3 - REVIEW SENTIMENT ANALYSIS")

X_train, X_test, y_train, y_test, label_map, y_train_original = prepare_sentiment_data(df)

print('Training reviews:', len(X_train))
print('Testing reviews:', len(X_test))

print('\nTraining distribution:')
print(np.bincount(y_train))

print('\nTesting distribution:')
print(np.bincount(y_test))

max_words = 5000
max_length = 100


tokenizer = Tokenizer(num_words = max_words, oov_token = '<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen = max_length, padding = 'post')
X_test_pad = pad_sequences(X_test_seq, maxlen = max_length, padding = 'post')

print('X_train shape:', X_train_pad.shape)
print('X_test shape:', X_test_pad.shape)

fasttext_path = 'model/cc.en.300.vec'

fasttext = KeyedVectors.load_word2vec_format(fasttext_path, binary = False)

embedding_dim = 300
embedding_matrix = np.zeros((max_words, embedding_dim))

word_index = tokenizer.word_index
found_words = 0

for word, index in word_index.items():
    if index >= max_words:
        continue

    if word in fasttext:
        embedding_matrix[index] = fasttext[word]
        found_words += 1

print('\nEmbedding matrix shape:')
print(embedding_matrix.shape)

print('\nwords found in FastText',found_words)

joblib.dump(tokenizer,'model/sentiment_fasttext_tokenizer.pkl')


with mlflow.start_run(run_name = 'Sentiment_Baseline'):

    majority_class = np.bincount(y_train_original).argmax()
    baseline_prediction = np.full(len(y_test),majority_class)
    baseline_metrics = evaluate_classification(y_test, baseline_prediction)
    log_parameters({'task' : 'Review sentiment', 'model' : 'Majority class Baseline'})
    log_metrics(baseline_metrics)
    print('\nSentiment Baseline')
    print_metrics(baseline_metrics)




sentiment_models = [{'name': 'FastText_LSTM_128_Dropout01','units': 128,'dropout': 0.1,'epochs': 30,'batch_size': 32},
                    {'name': 'FastText_LSTM_128_Dropout02','units': 128,'dropout': 0.2,'epochs': 30,'batch_size': 32},
                    {'name': 'FastText_LSTM_128_Dropout03','units': 128,'dropout': 0.3,'epochs': 30,'batch_size': 32},
                    {'name': 'FastText_LSTM_128_Dropout04','units': 128,'dropout': 0.2,'epochs': 40,'batch_size': 32},
                    {'name': 'FastText_LSTM_64_Dropout05','units': 64,'dropout': 0.2,'epochs': 30,'batch_size': 32}]

for i in sentiment_models:

    print('\nTraining:', i['name'])

    with mlflow.start_run(run_name = i['name']):

        model = Sequential([Embedding(input_dim = max_words, output_dim = embedding_dim,weights = [embedding_matrix],trainable = False),
                        LSTM(i['units']),
                        Dropout(i['dropout']),
                        Dense(3,activation = 'softmax')])

        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

        model.fit(X_train_pad, y_train,epochs = i['epochs'], batch_size = i['batch_size'],validation_split = 0.2)

        y_pred_prob = model.predict(X_test_pad)

        y_pred = np.argmax(y_pred_prob, axis = 1)

        metrics = evaluate_classification(y_test, y_pred)

        parameters = {'task': 'Review Sentiment','model': 'LSTM','embedding_dim': embedding_dim,'units': i['units'],
                      'dropout': i['dropout'],'epochs': i['epochs'],'batch_size': i['batch_size'],'max_words': max_words,
                      'max_length': max_length}

        log_parameters(parameters)
        log_metrics(metrics)
        log_model(model,i['name'])
        print_metrics(metrics)

        print('\nActual distribution:')
        print(np.bincount(y_test))

        print('\nPredicted distribution:')
        print(np.bincount(y_pred))

        print('\nUnique predictions:',np.unique(y_pred))


print('\nALL TASKS COMPLETED SUCCESSFULLY')
print('\n' + '-'*60)
print('\nMLflow tracking completed.')