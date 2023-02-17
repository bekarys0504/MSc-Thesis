import glob
import logging
from pathlib import Path

import antropy as ant
import click
import nolds
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from scipy import stats
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn interim data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('Building features')

    
    all_files = glob.glob(input_filepath+'/dataset_1_cheb2/*.csv', recursive=True)
    all_files = [x for x in all_files if 'post' not in x] # get only pre data
    all_files = all_files[:int(0.8*len(all_files))]
    all_files_test = all_files[int(0.8*len(all_files)):]

    SEGMENT_LENS = config['epochs'][2:3]
    CHANNELS = config['channels'][1:2]
    fs = config['freq_sampling']

    for channels in CHANNELS:
        for segment_len in SEGMENT_LENS:
            
            # create dataframe to store all features
            column_names = get_columns(channels, config)
            features_df = pd.DataFrame(columns=column_names)
            
            for file in all_files_test:
                subject_features_df = pd.DataFrame(columns=column_names)
                data = pd.read_csv(file)

                if 'PC1' in channels:
                    pca = PCA(n_components=len(channels))
                    pca.fit(data[config['ch_names_2']])
                    data = pd.DataFrame(pca.transform(data[config['ch_names_2']]), columns=channels)
                else:
                    data = data[channels]
                subject_class = [0 if 'healthy' in file else 1]

                segments = split_into_segments(data, segment_len, fs)

                for i, segment in enumerate(segments.values()):
                    features = extract_features(segment, channels, fs, config)
                    subject_features_df.loc[len(subject_features_df)] = features+subject_class

                features_df = features_df.append(subject_features_df)

            print('Saving pre_{}_ch_{}s_features.csv'.format(len(channels), segment_len))
            features_df.to_csv(output_filepath+'/Dataset_1/cheb_2/pre_test_{}_ch_{}s_features.csv'.format(len(channels), segment_len))

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

def extract_features(data_df, channels, fs, config):
    features = []

    #linear features
    features = features + [np.mean(np.abs(data_df[ch])) for ch in channels] # absolute mean of each channel
    features = features + [np.var(data_df[ch]) for ch in channels] # variance of each channel
    #features = features + [max(data_df[ch])  for ch in channels] # peaks
    features = features + [stats.skew(data_df[ch])  for ch in channels] # skewness
    features = features + [stats.kurtosis(data_df[ch])  for ch in channels] # kurtosis
    features = features + [np.sum(np.abs(data_df[ch])**2) for ch in channels] # energy

    hjorth_mob = []
    hjorth_comp = []
    band_powers = []

    # Define EEG bands
    eeg_bands = config['eeg_bands']

    for ch in channels:
        fft_vals = np.abs(rfft(data_df[ch].values))
        fft_freq = rfftfreq(len(data_df[ch].values), 1.0/fs)

        for band in eeg_bands:
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
            mean_power = np.mean(fft_vals[freq_ix ]**2)/fft_vals.size
            band_powers.append(mean_power)
            
        mob_val, comp_val = ant.hjorth_params(data_df[ch])
        hjorth_mob.append(mob_val) # Hjorth mobility
        hjorth_comp.append(comp_val) # Hjorth complexity
    features = features + hjorth_mob+hjorth_comp+band_powers

    #nonlinear features
    features = features + [nolds.hurst_rs(data_df[ch]) for ch in channels] # Hurst exponent
    features = features + [ant.sample_entropy(data_df[ch]) for ch in channels] # sample entropy
    features = features + [ant.higuchi_fd(data_df[ch]) for ch in channels] # Higuchi fractional dimension
    features = features + [ant.katz_fd(data_df[ch]) for ch in channels] # Katz fractional dimension
    features = features + [ant.spectral_entropy(data_df[ch],sf=fs,method='welch',normalize=True) for ch in channels]
    features = features + [ant.detrended_fluctuation(data_df[ch]) for ch in channels] # detrended fluctuation analysis

    return features

def get_columns(ch_names, config):
    eeg_bands = config['eeg_bands']

    mean_names = ["Mean-" + chan for chan in ch_names]
    var_names = ["Var-" + chan for chan in ch_names]
    #peak_names = ["Peak-" + chan for chan in ch_names]
    skew_names = ["Skewness-" + chan for chan in ch_names]
    kurt_names = ["Kurtosis-" + chan for chan in ch_names]
    energy_names = ["Energy-" + chan for chan in ch_names]
    hjmob_names = ["HjMob-" + chan for chan in ch_names]
    hjcomp_names = ["HjComp-" + chan for chan in ch_names]

    band_names = []
    for chan in ch_names:
        for bands in  eeg_bands:
            band_names.append(bands+'-'+chan)

    hurst_names = ["Hurst-" + chan for chan in ch_names]
    se_names = ["SampleEnt-" + chan for chan in ch_names]
    hfd_names = ["HFD-" + chan for chan in ch_names]
    kfd_names = ["KFD-" + chan for chan in ch_names]
    pse_names = ["Pse-" + chan for chan in ch_names]
    dfa_names = ["DFA-" + chan for chan in ch_names]


    column_names = mean_names+var_names+skew_names+kurt_names+energy_names+hjmob_names+hjcomp_names+band_names+hurst_names+se_names+hfd_names+kfd_names+pse_names+dfa_names+['depressed']
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