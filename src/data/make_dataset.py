# -*- coding: utf-8 -*-
import glob
import logging
from pathlib import Path

import chart_studio.plotly as py
import click
import h5io
import kaleido
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import openpyxl
import pandas as pd
import PyQt5
from autoreject import get_rejection_threshold
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from plotly.graph_objs import Annotations, Figure, Layout, Marker, Scatter
from plotly.graph_objs.layout import Annotation, Font, YAxis
import os

mne.set_log_level(False)
# I had error as in this link https://github.com/open-mmlab/mmdetection/issues/7035

#matplotlib.use('Agg')
matplotlib.use('Qt5Agg')
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
    extension = '.npy'
    layout = read_montage()
    all_files = glob.glob(input_filepath+'/Dataset_1/All Participants//**/*.edf', recursive=True)
    all_files = remove_noisy_data(all_files, config)
    #all_files = all_files[:2]
    df = pd.DataFrame(columns=['fname', 'channels', 'length', 'ic_removed'])

    # iterate through all .edf files in 'input_filepath'
    for files in all_files:
        new_filename = get_new_filename(files)
        print(new_filename)
        filename = Path(files).stem # filename without extension
        # read files
        raw = mne.io.read_raw_edf(files, preload=True);
        raw = rename_channels(raw)
        raw.set_montage(layout, on_missing = 'ignore');
        raw_filtered = bandpass_filter(raw)
        
        #plot_eeg_data(raw_filtered, filename, 500, False, './reports/figures/eeg/raw/', True)
        #raw_filtered.plot()
        #raw_ica, excluded_channels = perform_ica(raw_filtered)
        #raw_ica.plot()
        
        raw_ica = raw_filtered.copy()

        #plot_eeg_data(raw_ica, filename, 500, False, './reports/figures/eeg/raw/', True)
        #mne.export.export_raw(output_filepath+new_filename+'.edf', raw_ica, overwrite=True)
        if os.path.exists(output_filepath+'/'+new_filename+extension):
            print('file exists')
            new_filename = new_filename+'_1'+extension
        
        np.save(output_filepath+'/'+new_filename+extension, raw_ica.get_data)

        #df = df.append(pd.Series({'fname':filename, 'channels':raw_filtered.get_data().shape[0], 'length':raw_filtered.get_data().shape[1], 'ic_removed':excluded_channels}), ignore_index=True)
        
        # plot eeg data and it's power spectral density and save them in figures
        if plot_eeg:
            plot_eeg_data(raw, filename, 500, save=save_eeg, save_folder='./reports/figures/eeg/raw/')
            plot_eeg_data(raw_filtered, filename, 500, save=save_eeg, save_folder='./reports/figures/eeg/filtered/')
        if plot_psd:
            plot_psd_data(raw, config, filename, save=save_psd, save_folder='./reports/figures/psd/raw/')
            plot_psd_data(raw_filtered, config, filename, save=save_psd, save_folder='./reports/figures/psd/filtered/')

    if save_excel_file:
        df.to_excel('./reports/data.xlsx')

def get_new_filename(path):
    subject = get_subject_number(path)
    eye_state = get_eye_state(path)
    state = check_state(path)
    subject_class = get_class(subject)

    new_filename = subject+'_'+subject_class+'_'+state+'_'+eye_state
    return new_filename

def get_subject_number(path):
    return path.split('\\')[1][:-1]

def get_class(subject):
    healthy = config['healthy_subjects']
    depressed = config['depressed_subjects']

    if subject in healthy:
        return "healthy"
    elif subject in depressed:
        return "depressed"  
    else:
        return 'NOT_FOUND'

def get_eye_state(string):
    if 'ec' in string.lower():
        return "ec"
    elif 'eo' in string.lower():
        return "eo"
    else:
        return 'NOT_FOUND'   

def check_state(string):
    if 'pre' in string.lower():
        return 'pre'
    elif 'post' in string.lower():
        return 'post'
    else:
        return 'NOT_FOUND'

def plot_psd_data(data, config, filename, save, save_folder):
    fig, axes = plt.subplots(figsize = (8,3));
    data.plot_psd(ax=axes, fmax=100, color = config.colors['dtu_red'], show=False, spatial_colors=False);
    
    if save:
        fig.savefig(save_folder + filename +'.png') # save figures     

def plot_eeg_data(data, filename, fs, save, save_folder, show=False, plot_title=None):
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
    
    if plot_title is not None:  
        fig.update_layout(
            title=plot_title,
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
    if show:
        fig.show()
    if save:
        fig.write_image(save_folder + filename +'.png') 

def bandpass_filter(data):
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
    
    # ICA parameters
    random_state = 42   # ensures ICA is reproducable each time it's run
    ica_n_components = 31     # Specify n_components as a decimal to set % explained variance

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=ica_n_components,
                                random_state=random_state,
                                )
    ica.fit(epochs_ica,
            reject=reject,
            tstep=tstep)

    ica.plot_sources(inst=data, show=True)
    ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': 40});

    exclude_channels = input("Input:")
    if len(exclude_channels) > 0:
        exclude_channels = [np.int64(v) for v in exclude_channels.split(",")]
    
    ica.exclude = exclude_channels
    ica.apply(data)

    return data, exclude_channels

def rename_channels(data):
    """
    rename channels to remove 'EEG', '-A1' and '-A2'
    """
    ch_names_dict = {}
    for i in range(31):
        ch_name = data.info['chs'][i]['ch_name']
        ch_name_new = ch_name
        ch_name_new = ch_name_new.replace('EEG ', '')
        ch_name_new = ch_name_new.replace('-A1', '')
        ch_name_new = ch_name_new.replace('-A2', '')
        ch_names_dict[ch_name] = ch_name_new
    
    mne.rename_channels(data.info, ch_names_dict)
    return data

def remove_noisy_data(filelist, config):
    noisy_recs = config['noisy_recs']
    filelist = [x for x in filelist if Path(x).stem not in noisy_recs]
    return filelist

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
