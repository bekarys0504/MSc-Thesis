# Importing Necessary modules
from fastapi import FastAPI, File, UploadFile
import uvicorn
import pandas as pd
from omegaconf import OmegaConf
import pickle
from src.features.build_features import get_columns, split_into_segments, extract_features
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras

config = OmegaConf.load('./config/config.yaml')
ml_model_path = r'./models/ml_models/'
cnn_model_path = r'.../models/weights/exp31/'
# Declaring our FastAPI instance
app = FastAPI()

@app.post("/")
def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    file.file.close()
    segment_len = 10
    classes = {0: 'Healthy', 1: 'Depressed'}

    X, y, features_df = get_data(df, file.filename, segment_len, ml_model_path)
    
    KNN_pred, SVM_pred, RF_pred, XGBoost_pred = predict_ml(X, ml_model_path)
    #CNN_pred = predict_cnn(X, cnn_model_path)

    return {"filename": file.filename, 
            'KNN prediction': classes[KNN_pred], 
            'SVM prediction': classes[SVM_pred], 
            'RF prediction': classes[RF_pred], 
            'XGBoost prediction': classes[XGBoost_pred], 
            'CNN prediction': classes[XGBoost_pred]}


def get_data(data_df, filename, segment_len, ml_model_path):
    channels = config['predict_19_ch']
    fs = config['freq_sampling']
    column_names = get_columns(channels, config)
    features_df = pd.DataFrame(columns=column_names)

    subject_class = [0 if 'healthy' in filename else 1]
    segments = split_into_segments(data_df, segment_len, fs)
    for i, segment in enumerate(segments.values()):
        features = extract_features(segment, channels, fs, config)
        features_df.loc[len(features_df)] = features+subject_class

    # drop rows with null values
    features_df.dropna(inplace=True)
    scaler = pickle.load(open(ml_model_path+'scaler.pkl', 'rb'))
    data_scaled = scaler.transform(features_df)

    X = data_scaled[:,:-1] # get all features
    y = data_scaled[:,-1] # get labels

    return X, y, features_df

def get_cnn_data(filename, s_data, epoch_length):
    X = np.array([])
    y = []
    start = 0
    end = epoch_length*500
    step = int((end-start))
    count = 0
    while end < len(s_data):
        if 'depressed' in filename.lower():
            # normalize
            s_epoch = s_data[start:end]

            if 0 not in s_epoch.std().values:
                s_epoch = (s_epoch-s_epoch.mean())/s_epoch.std()
                X = np.append(X, s_epoch.values.astype(np.double))
                y.append(1)

        elif 'healthy' in filename.lower():
            # normalize
            s_epoch = s_data[start:end]

            if 0 not in s_epoch.std().values:
                s_epoch = (s_epoch-s_epoch.mean())/s_epoch.std()
                X = np.append(X, s_epoch.values.astype(np.double))
                y.append(0)

            
        count += 1
        start = start+step
        end = end+step

def predict_ml(X, ml_model_path):

    # predict with ml models
    models = ['KNN.pkl', 'SVM.pkl', 'RF.pkl', 'XGBoost.pkl']
    predictions = []

    for ml_model in models:
        # load models and do prediction
        with open(ml_model_path+ml_model, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X)

        most_frequent_class = np.argmax(np.bincount(y_pred.astype(int)))
        predictions.append(most_frequent_class)
    
    return predictions

def predict_cnn(X, cnn_model_path):
    # predict with CNN model
    # load model
    model = keras.models.load_model(cnn_model_path+'best.hdf5')
    test_predictions = model.predict(x = X)
    predictions = np.argmax(test_predictions, axis=1)
    return predictions
    

