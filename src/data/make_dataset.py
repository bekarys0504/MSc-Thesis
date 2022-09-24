# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import pandas as pd
import openpyxl
import kaleido

import click
import matplotlib
import matplotlib.pyplot as plt
import mne
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
import glob
from pathlib import Path
from autoreject import get_rejection_threshold

from plotly import tools
import chart_studio.plotly as py
from plotly.graph_objs import Layout, Scatter, Figure, Marker, Annotations
from plotly.graph_objs.layout import YAxis, Font, Annotation

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
    plot_eeg = False
    plot_psd = False
    save_eeg = False
    save_psd = False
    layout = read_montage()
    all_files = glob.glob('./data/raw/Dataset_1/All Participants//**/*.edf', recursive=True)
    config = OmegaConf.load('./config/config.yaml')

    df = pd.DataFrame(columns=['fname', 'channels', 'length'])

    # iterate through all .edf files in 'input_filepath'
    for files in all_files:
        filename = Path(files).stem # filename without extension
        # read files
        raw = mne.io.read_raw_edf(files, preload=True);
        raw.set_montage(layout, on_missing = 'ignore');
        
        raw_filtered = bandpass_filter(raw, config)

        df = df.append(pd.Series({'fname':filename, 'channels':raw_filtered.get_data().shape[0], 'length':raw_filtered.get_data().shape[1]}), ignore_index=True)
        
        # plot eeg data and it's power spectral density and save them in figures
        if plot_eeg:
            plot_eeg_data(raw, config, filename, 500, save=save_eeg, save_folder='./reports/figures/eeg/raw/')
            plot_eeg_data(raw_filtered, config, filename, 500, save=save_eeg, save_folder='./reports/figures/eeg/filtered/')
        if plot_psd:
            plot_psd_data(raw, config, filename, save=save_psd, save_folder='./reports/figures/psd/raw/')
            plot_psd_data(raw_filtered, config, filename, save=save_psd, save_folder='./reports/figures/psd/filtered/')

    if save_excel_file:
        df.to_excel('./reports/data.xlsx')

def plot_psd_data(data, config, filename, save, save_folder):
    fig, axes = plt.subplots(figsize = (8,3));
    data.plot_psd(ax=axes, fmax=100, color = config.colors['dtu_red'], show=False, spatial_colors=False);
    
    if save:
        fig.savefig(save_folder + filename +'.png') # save figures     

def plot_eeg_data(data, config, filename, fs, save, save_folder, show=False):
    picks = mne.pick_types(data.info, eeg=True, exclude=[])
    #start, stop = data.time_as_index([0, data.n_times/fs])
    start, stop = data.time_as_index([0, 20])

    n_channels = 31
    ch_names = [data.info['ch_names'][p] for p in picks[:n_channels]]
    data, times = data[picks[:n_channels], start:stop]

    step = 1. / n_channels
    kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)

    # create objects for layout and traces
    layout = Layout(yaxis=YAxis(kwargs), showlegend=False)
    traces = [Scatter(x=times, y=data.T[:, 0],  line=dict(width=0.5))]

    # loop over the channels
    for ii in range(1, n_channels):
            kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
            layout.update({'yaxis%d' % (ii + 1): YAxis(kwargs), 'showlegend': False})
            traces.append(Scatter(x=times, y=data.T[:, ii], yaxis='y%d' % (ii + 1),  line=dict(width=0.5)))

    # set the size of the figure and plot it
    layout.update(autosize=False, width=1500, height=1000)
    fig = Figure(data=traces, layout=layout)
    fig.update_xaxes(anchor='free')
    fig.update_traces(marker=dict(
            color='black'))

    for ii, ch_name in enumerate(ch_names):
            fig.add_annotation(x=-0.04, y=0, xref='paper', yref='y%d' % (ii + 1), text=ch_name, font=dict(size=9), showarrow=False)
    
    if show:
        fig.show()
    if save:
        fig.write_image(save_folder + filename +'.png') 

def bandpass_filter(data, config):
    low_cut = config.preprocess['low_cut']
    high_cut  = config.preprocess['high_cut']

    data_filt = data.copy().filter(low_cut, high_cut);

    return data_filt

def read_montage(show=False):
    layout = pd.read_csv(r'.\data\raw\Dataset_1\Standard-10-10-Cap31-eeg.txt', sep = '\t')
    layout.columns = layout.columns.str.strip()
    layout["labels"] = layout["labels"].str.strip()
    layout = layout.set_index('labels')
    layout = layout.to_dict(orient = "index")
    for channel in layout.keys():
        yxz = np.array([layout[channel]["Y"]/15, layout[channel]["X"]/15, layout[channel]["Z"]/15])
        layout[channel] = yxz
    layout = mne.channels.make_dig_montage(layout, coord_frame='head')
    if show:
        mne.viz.plot_montage(layout);
    
    return layout

def perform_ica(data, step=1):
    # Break raw data into 1 s epochs
    tstep = step
    events_ica = mne.make_fixed_length_events(data, duration=tstep);
    epochs_ica = mne.Epochs(data, events_ica,
                            tmin=0.0, tmax=tstep,
                            baseline=None,
                            preload=True);

    reject = get_rejection_threshold(epochs_ica);

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
