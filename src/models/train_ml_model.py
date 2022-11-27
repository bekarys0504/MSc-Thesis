import os
import numpy as np
import random
import logging
import click
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from pathlib import Path
import glob
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SequentialFeatureSelector as seqfs
from genetic_selection import GeneticSelectionCV
from mrmr import mrmr_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import statistics
import pickle

os.environ['PYTHONHASHSEED'] = '0'
config = OmegaConf.load('./config/config.yaml')

# set randomm seed for reproducability
np.random.seed(config['random_seed'])
random.seed(config['random_seed'])

@click.command()

def main():

    #all_feature_files = glob.glob('./data/processed/Dataset_1/chs_new/pre_19_ch_10s_features.csv', recursive=True)
    filename = 'pre_19_ch_10s_features.csv'
    all_feature_files = ['./data/processed/Dataset_1/cheb_2/'+filename]

    feature_selectors = [anova_fs]
    models = [KNeighborsClassifier(n_neighbors=2), svm.SVC(kernel='poly'), RandomForestClassifier(), XGBClassifier()]
    percentages = config['feature_percentages']
    n_folds = config['n_folds']
    RANDOM_SEED = config['random_seed']

    for file in all_feature_files:
        X, y, features_df = get_data(file)

        for percent in percentages:
                for fs in feature_selectors:
                        print('filename: '+file+' FS: ' + str(fs)[10:-22] + 'percentage: ' + str(percent))

                        if (('genetic' not in str(fs)) and ('forward' not in str(fs))) and percent != 'all':
                                X_sel, selected_feature_names = fs(features_df, X, y, percent)
                                print('Number of features selected '+str(X_sel.shape[1])+' out of '+str(X.shape[1]))

                        for model in models:

                                if percent == 'all':
                                        print('using all features')
                                        selected_feature_names = features_df.columns[1:]
                                        X_sel = X
                                else:
                                        if 'genetic' in str(fs) or 'forward' in str(fs):
                                                X_sel, selected_feature_names = fs(features_df, X, y, model, percent)
                                                print('Number of features for genetic or forward '+str(X_sel.shape[1])+' out of '+str(X.shape[1]))

                                print_cv_results(model, X_sel, y, n_folds)
                                X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.33, random_state=RANDOM_SEED, stratify=y)
                                
                                model.fit(X_train, y_train)
                                
                                # save
                                save_model(model)

                                y_pred = model.predict(X_test)
                                print('Test accuracy on Dataset 1:', accuracy_score(y_test, y_pred))
                                # test on Dataset 2
                                #X_dataset2, y_dataset2, features_df2 = get_data('./data/processed/Dataset_2/'+filename, selected_feature_names)
                                
                                #test_model(model, X_dataset2, y_dataset2)


def save_model(model):

    # save KNN model
    if 'kneighbors' in str(model).lower():
        with open(r'./models/ml_models/'+'KNN.pkl','wb') as f:
            pickle.dump(model,f)

    # save svm model
    if 'svc' in str(model).lower():
        with open(r'./models/ml_models/'+'SVM.pkl','wb') as f:
            pickle.dump(model,f)

    # save RF model
    if 'randomforest' in str(model).lower():
        with open(r'./models/ml_models/'+'RF.pkl','wb') as f:
            pickle.dump(model,f)

    # save XGBoost model
    if 'xgbclassifier' in str(model).lower():
        with open(r'./models/ml_models/'+'XGBoost.pkl','wb') as f:
            pickle.dump(model,f)

def test_model(model, X, y):
    accuracy, precision, recall, f1_score = [], [], [], [] 

    for _ in range(config['n_folds']):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        y_pred = model.predict(X_test)
        precision_val, recall_val, f1_score_val, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        accuracy.append(accuracy_score(y_test, y_pred)*100)
        precision.append(precision_val*100)
        recall.append(recall_val*100)
        f1_score.append(f1_score_val*100)
        
    print(str(model)+" accuracy: %0.2f±%0.2f    precison: %0.2f±%0.2f   recall: %0.2f±%0.2f     f1_score: %0.2f±%0.2f" 
            % (statistics.mean(accuracy), statistics.stdev(accuracy), statistics.mean(precision), statistics.stdev(precision), 
            statistics.mean(recall), statistics.stdev(recall), statistics.mean(f1_score), statistics.stdev(f1_score)))

def get_data(filepath, selected_feature_names = None):
    features_df = pd.read_csv(filepath).iloc[: , 1:]

    if selected_feature_names is not None:
        features_df = features_df[selected_feature_names.to_list()+['depressed']]
    # drop rows with null values
    features_df.dropna(inplace=True)
    #features_df = features_df[features+['depressed']]
    scaler = MinMaxScaler() 
    data_scaled = scaler.fit_transform(features_df)

    X = data_scaled[:,:-1] # get all features
    y = data_scaled[:,-1] # get labels

    return X, y, features_df

def anova_fs(features_df, X, y, percentage):
    num_features = int(X.shape[1]*percentage/100) 

    fvalue_Best = SelectKBest(f_classif, k=num_features)
    X_sel = fvalue_Best.fit_transform(X, y)

    feature_indices = fvalue_Best.get_support(indices=True)
    selected_feature_names = features_df.iloc[:,:-1].columns[feature_indices] 

    return X_sel, selected_feature_names

def forward_fs(features_df, X, y, model, percentage):
    num_features = int(X.shape[1]*percentage/100) 
    sfs = seqfs(model, n_features_to_select=num_features, direction='forward')
    sfs.fit(X, y)

    feature_indices = sfs.get_support(indices=True)
    selected_feature_names = features_df.iloc[:,:-1].columns[feature_indices] 
    
    X_sel = sfs.transform(X)
    return X_sel, selected_feature_names

def genetic_algorithm_fs(features_df, X, y, model, percentage):

    num_features = int(X.shape[1]*percentage/100) 
    models = GeneticSelectionCV(
        model, cv=5, verbose=0,
        scoring="accuracy", max_features=num_features,
        n_population=50, crossover_proba=0.8,
        mutation_proba=0.1, n_generations=100,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.04,
        tournament_size=3, n_gen_no_change=10,
        caching=True, n_jobs=-1)

    models = models.fit(X, y)
    selected_feature_names = features_df.iloc[:,:-1].columns[models.support_]

    X_sel = X[:,models.support_]
    return X_sel, selected_feature_names

def mrmr_fs(features_df, X, y, percentage):

    num_features = int(X.shape[1]*percentage/100) 
    selected_features = mrmr_classif(pd.DataFrame(X), pd.Series(y), K = num_features, show_progress=False)
    X_sel = X[:, selected_features]
    
    selected_feature_names = features_df.iloc[:,:-1].columns[selected_features]

    return X_sel, selected_feature_names

def print_cv_results(model, X, y, n_folds):
    
    accuracy = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=n_folds, shuffle = True)) *100
    precision = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=n_folds, shuffle = True), scoring='precision') *100
    recall = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=n_folds, shuffle = True), scoring='recall') *100
    f1_score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=n_folds, shuffle = True), scoring='f1') *100
    
    print(str(model)+" accuracy: %0.2f±%0.2f    precison: %0.2f±%0.2f   recall: %0.2f±%0.2f     f1_score: %0.2f±%0.2f" 
            % (accuracy.mean(), accuracy.std(), precision.mean(), precision.std(), recall.mean(), recall.std(), f1_score.mean(), f1_score.std()))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()