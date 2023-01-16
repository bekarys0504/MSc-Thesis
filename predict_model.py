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
import sys
import math
sys.path.insert(1, './src/models')
from CustomEnsembleModel import CustomEnsembleModel
import time

config = OmegaConf.load('./config/config.yaml')
ml_model_path = r'./models/ml_models/'
cnn_model_path = r'./models/weights/exp37/'
# Declaring our FastAPI instance
app = FastAPI()

@app.post("/")
def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    file.file.close()
    segment_len = 10
    cnn_segment_len = 1
    classes = {0: 'Healthy', 1: 'Depressed'}
    st = time.time()
    X, y, features_df = get_data(df, file.filename, segment_len, ml_model_path)
    predictions, confidences = predict_ml(X, ml_model_path)
    et = time.time()
    elapsed_time = et - st
    print('ML models execution time:', elapsed_time, 'seconds')
    
    st = time.time()
    X_cnn, y_cnn = get_cnn_data(file.filename, df, cnn_segment_len)
    CNN_pred, CNN_confidence = predict_cnn(X_cnn, cnn_model_path)
    et = time.time()
    elapsed_time = et - st
    print('CNN execution time:', elapsed_time, 'seconds')

    return {"filename": file.filename, 
            'KNN prediction': str(classes[predictions[0][0]])+str(' (')+str(confidences[0])+str('%)'),
            'SVM prediction': str(classes[predictions[1][0]])+str(' (')+str(confidences[1])+str('%)'),
            'RF prediction': str(classes[predictions[2][0]])+str(' (')+str(confidences[2])+str('%)'),
            'XGBoost prediction': str(classes[predictions[3][0]])+str(' (')+str(confidences[3])+str('%)'),
            'Ensemble prediction': str(classes[predictions[4][0]])+str(' (')+str(confidences[4])+str('%)'),
            'CNN prediction': str(classes[CNN_pred])+str(' (')+str(CNN_confidence)+str('%)')}


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
    s_data = s_data[config['ch_names']] # select channels
    X = []
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
                X.append(s_epoch.values.astype(np.double))
                y.append(1)

        elif 'healthy' in filename.lower():
            # normalize
            s_epoch = s_data[start:end]

            if 0 not in s_epoch.std().values:
                s_epoch = (s_epoch-s_epoch.mean())/s_epoch.std()
                X.append(s_epoch.values.astype(np.double))
                y.append(0)

            
        count += 1
        start = start+step
        end = end+step
    
    return np.array(X), y

def predict_ml(X, ml_model_path):

    # predict with ml models
    models = ['KNN.pkl', 'SVM.pkl', 'RF.pkl', 'XGBoost.pkl', 'CustomEnsembleModel.pkl']
    predictions = []
    confidences = []

    for ml_model in models:
        # load models and do prediction
        with open(ml_model_path+ml_model, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        predicted_class, model_confidence = combine_using_Dempster_Schafer(np.expand_dims(y_pred_proba[:,1], axis=0))
        
        #most_frequent_class = np.argmax(np.bincount(y_pred.astype(int)))
        predictions.append(predicted_class)
        confidences.append(model_confidence)
    return predictions, confidences

def predict_cnn(X, cnn_model_path):
    # predict with CNN model
    # load model
    
    model = keras.models.load_model(cnn_model_path+'best.hdf5')
    test_predictions = model.predict(x = X)
    confidence = np.max(np.mean(test_predictions, axis=0))*100
    predictions = np.argmax(test_predictions, axis=1)
    return np.argmax(np.bincount(predictions.astype(int))), confidence
    
def combine_using_Dempster_Schafer(p_individual):
    bpa0 = 1.0 - np.prod(p_individual, axis=1)
    bpa1 = 1 - np.prod(1 - p_individual, axis=1)
    belief = np.vstack([bpa0 / (1 - bpa0), bpa1 / (1 - bpa1)]).T #B

    if np.any(np.isinf(belief)):
        confidence = 1.0
    else:
        normalized_belief = belief/belief.sum(axis=1,keepdims=1)
        confidence = np.max(normalized_belief)

    #print(belief/belief.sum(axis=0,keepdims=1))
    y_final = np.argmax(belief, axis=1) #C

    return y_final, confidence*100
