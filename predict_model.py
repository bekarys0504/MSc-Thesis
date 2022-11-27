# Importing Necessary modules
from fastapi import FastAPI, File, UploadFile
import uvicorn
import pandas as pd
from omegaconf import OmegaConf
import pickle
from src.features.build_features import get_columns, split_into_segments, extract_features
from sklearn.preprocessing import MinMaxScaler
import numpy as np

config = OmegaConf.load('./config/config.yaml')

# Declaring our FastAPI instance
app = FastAPI()

@app.post("/")
def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    file.file.close()
    segment_len = 10
    classes = {0: 'Healthy', 1: 'Depressed'}

    X, y, features_df = get_data(df, file.filename, segment_len)
    
    KNN_pred, SVM_pred, RF_pred, XGBoost_pred = predict(X)

    return {"filename": file.filename, 
            'KNN prediction': classes[KNN_pred], 
            'SVM prediction': classes[SVM_pred], 
            'RF prediction': classes[RF_pred], 
            'XGBoost prediction': classes[XGBoost_pred], 
            'CNN prediction': classes[XGBoost_pred]}


def get_data(data_df, filename, segment_len):
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
    scaler = MinMaxScaler() 
    data_scaled = scaler.fit_transform(features_df)

    X = data_scaled[:,:-1] # get all features
    y = data_scaled[:,-1] # get labels

    return X, y, features_df

def predict(X):
    models = ['KNN.pkl', 'SVM.pkl', 'RF.pkl', 'XGBoost.pkl']
    predictions = []

    for ml_model in models:
        # load models and do prediction
        with open(r'./models/ml_models/'+ml_model, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X)

        most_frequent_class = np.argmax(np.bincount(y_pred.astype(int)))
        predictions.append(most_frequent_class)

    return predictions
    

