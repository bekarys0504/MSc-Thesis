# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import pandas as pd
import openpyxl

import click
import matplotlib
import matplotlib.pyplot as plt
import mne
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf

# I had error as in this link https://github.com/open-mmlab/mmdetection/issues/7035
matplotlib.use('Agg')

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    save_excel_file = False
    dataset_path = input_filepath + "/Dataset_1/"
    config = OmegaConf.load('./config/config.yaml')

    df = pd.DataFrame(columns=['fname', 'channels', 'length'])

    # iterate through all .edf files in 'input_filepath'
    for files in os.listdir(dataset_path):
        if files.endswith('.edf'):

            # read files
            raw = mne.io.read_raw_edf(dataset_path + files, preload=True)
            raw_filtered = bandpass_filter(raw, config)

            df = df.append(pd.Series({'fname':files, 'channels':raw_filtered.get_data().shape[0], 'length':raw_filtered.get_data().shape[1]}), ignore_index=True)
            
            '''
            # plot eeg data and it's power spectral density and save them in figures
            plot_eeg(raw, config, files, save=True, save_folder='./reports/figures/eeg/raw/')
            plot_psd(raw, config, files, save=True, save_folder='./reports/figures/psd/raw/')

            plot_eeg(raw_filtered, config, files, save=True, save_folder='./reports/figures/eeg/filtered/')
            plot_psd(raw_filtered, config, files, save=True, save_folder='./reports/figures/psd/filtered/')
            ''' 

    if save_excel_file:
        df.to_excel('./reports/data.xlsx')

def plot_psd(data, config, filename, save, save_folder):
    fig, axes = plt.subplots(figsize = (8,3));
    data.plot_psd(ax=axes, fmax=100, color = config.colors['dtu_red'], show=False, spatial_colors=False);
    
    if save:
        fig.savefig(save_folder + filename[:-4] +'.png') # save figures by removing last 4 chars, which are '.edf'          

def plot_eeg(data, config, filename, save, save_folder):
    fig = mne.viz.plot_raw(data, show=False)
    
    if save:
        fig.savefig(save_folder + filename[:-4] +'.png') # save figures by removing last 4 chars, which are '.edf'   

def bandpass_filter(data, config):
    low_cut = config.preprocess['low_cut']
    high_cut  = config.preprocess['high_cut']

    data_filt = data.copy().filter(low_cut, high_cut)

    return data_filt

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
