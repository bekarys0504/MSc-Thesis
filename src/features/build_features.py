import click
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
import logging
from pathlib import Path
import glob
import pandas as pd
import numpy as np
from scipy import stats
import nolds
import antropy as ant

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn interim data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('Building features')

    
    all_files = glob.glob('./data/interim/dataset_1_cheb2/*.csv', recursive=True)
    all_files = [x for x in all_files if 'post' not in x] # get only pre data

    SEGMENT_LENS = config['epochs']
    CHANNELS = config['channels']
    fs = config['freq_sampling']

    for channels in CHANNELS:
        for segment_len in SEGMENT_LENS:
            
            # create dataframe to store all features
            column_names = get_columns(channels)
            features_df = pd.DataFrame(columns=column_names)
            
            for file in all_files:
                data = pd.read_csv(file)
                data = data[channels]
                subject_class = [0 if 'healthy' in file else 1]

                segments = split_into_segments(data, segment_len, fs)

                for i, segment in enumerate(segments.values()):
                    features = extract_features(segment, channels)
                    features_df.loc[len(features_df)] = features+subject_class
            
            print('Saving pre_{}_ch_{}s_features.csv'.format(len(channels), segment_len))
            
            features_df.to_csv(output_filepath+'/Dataset_1/cheb_2/pre_{}_ch_{}s_features.csv'.format(len(channels), segment_len))

# function to split data into segments
def split_into_segments(df,split_seg_len ,fs):
    sample_split_len = split_seg_len*fs # samples
    start_segments = np.arange(0, len(df), sample_split_len) # (start point, stop point, step length)
    i=0 # counter
    d = {} # empty
    for j in start_segments:
        i+=1
        d["segment{0}".format(i)] = df.iloc[j:j+sample_split_len ,:]
    
    return d

def extract_features(data_df, channels):
    features = []

    # normalize each channel
    # data_df=(data_df-data_df.min())/(data_df.max()-data_df.min())

    features = features + [np.mean(np.abs(data_df[ch])) for ch in channels] # absolute mean of each channel
    features = features + [np.var(data_df[ch]) for ch in channels] # variance of each channel
    features = features + [max(data_df[ch])  for ch in channels] # peaks
    features = features + [stats.skew(data_df[ch])  for ch in channels] # skewness
    features = features + [stats.kurtosis(data_df[ch])  for ch in channels] # kurtosis
    features = features + [np.sum(np.abs(data_df[ch])**2) for ch in channels] # energy
    features = features + [np.std(data_df[ch])  for ch in channels] # standard deviation
    features = features + [nolds.hurst_rs(data_df[ch]) for ch in channels] # Hurst exponent
    features = features + [ant.sample_entropy(data_df[ch]) for ch in channels] # sample entropy
    features = features + [ant.hjorth_params(data_df[ch])[0] for ch in channels] # Hjorth mobility
    features = features + [ant.hjorth_params(data_df[ch])[1] for ch in channels] # Hjorth complexity

    return features

def get_columns(ch_names):
    mean_names = ["Mean-" + chan for chan in ch_names]
    var_names = ["Var-" + chan for chan in ch_names]
    peak_names = ["Peak-" + chan for chan in ch_names]
    skew_names = ["Skewness-" + chan for chan in ch_names]
    kurt_names = ["Kurtosis-" + chan for chan in ch_names]
    energy_names = ["Energy-" + chan for chan in ch_names]
    std_names = ["Std-" + chan for chan in ch_names]
    hurst_names = ["Hurst-" + chan for chan in ch_names]
    se_names = ["SampleEnt-" + chan for chan in ch_names]
    hjmob_names = ["HjMob-" + chan for chan in ch_names]
    hjcomp_names = ["HjComp-" + chan for chan in ch_names]


    column_names = mean_names+var_names+peak_names+skew_names+kurt_names+energy_names+std_names+hurst_names+se_names+hjmob_names+hjcomp_names+['depressed']
    return column_names


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    config = OmegaConf.load('./config/config.yaml')
    main()