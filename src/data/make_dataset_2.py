# -*- coding: utf-8 -*-
import glob
import logging
import os
import random
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
from scipy import signal, stats

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
    logger.info('making final data set from raw data 2')

    save_excel_file = False
    plot_eeg = False
    plot_psd = False
    save_eeg = False
    save_psd = False
    extension = '.csv'
    dataset = 'Dataset2'
    layout = read_montage()

    if dataset == 'Dataset1':
        all_files = glob.glob(input_filepath+'/Dataset_1/All Participants//**/*.edf', recursive=True)
    if dataset == 'Dataset1_1':
        all_files = glob.glob(input_filepath+'/Dataset_1/**//**/*.set', recursive=True)
    if dataset == 'Dataset2':
        all_files = glob.glob(input_filepath+'/Dataset_2/*.edf', recursive=True)
        
    all_files = remove_noisy_data(all_files, config)
    random.shuffle(all_files)
    df = pd.DataFrame(columns=['fname', 'channels', 'length', 'ic_removed'])
    
    # iterate through all .edf files in 'input_filepath'
    for files in all_files:
        new_filename, subject, state, eye_state, subject_class = get_new_filename(files)

        filename = Path(files).stem # filename without extension

        # read files
        if dataset == 'Dataset1' or dataset == 'Dataset2':
            raw = mne.io.read_raw_edf(files, preload=True);
        if dataset == 'Dataset1_1':
            raw = mne.io.read_raw_eeglab(files, preload=True);
        
        #raw = rename_channels(raw)
        raw_filtered = filter_cheb2(raw)
        
        # perform ica
        raw_filtered.plot()
        raw_ica, excluded_channels = perform_ica(raw_filtered.copy())
        raw_filtered.plot()
        raw_ica.plot()
        
        # save files
        if os.path.exists(output_filepath+'/'+new_filename+extension):
            print('file exists')
            new_filename = new_filename+'_1'+extension
        
        data_df = raw_ica.to_data_frame(picks=config['ch_names_2'])
        data_df['subject'] = subject
        data_df['post_pre'] = state
        data_df['eye_state'] = eye_state
        data_df['class'] = subject_class
        data_df.to_csv(output_filepath+'/'+new_filename+extension)

        df = df.append(pd.Series({'fname':filename, 'channels':raw_filtered.get_data().shape[0], 'length':raw_filtered.get_data().shape[1], 'ic_removed':excluded_channels}), ignore_index=True)
        

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
    state = 'PRE'
    subject_class = get_class(path)

    new_filename = subject+'_'+subject_class+'_'+state+'_'+eye_state
    return new_filename, subject, state, eye_state, subject_class

def get_subject_number(path):
    return path.split(' ')[1]

def get_class(path):
    healthy = config['healthy_subjects']
    depressed = config['depressed_subjects']

    if 'H' in path:
        return "healthy"
    elif 'MDD' in path:
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
    data.plot_psd(ax=axes, fmax=100, color = config.colors['dtu_red'], show=False, spatial_colors=False, exclude=['event', 'bads']);
    
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

def filter(data):
    # crop the signal
    max_seconds = (data.get_data().shape[1]-1)/500 # calculate total time
    data.crop(tmin=1, tmax=max_seconds-2) # crop end

    low_cut = config.preprocess['low_cut']
    high_cut  = config.preprocess['high_cut']

    data_filt = data.copy().filter(low_cut, high_cut);
    data_filt = data_filt.copy().notch_filter(50);
    return data_filt

def filter_cheb2(data):
    # crop the signal
    max_seconds = (data.get_data().shape[1]-1)/500 # calculate total time
    data.crop(tmin=1, tmax=max_seconds-2) # crop end

    fs = config['freq_sampling']
    BP_EDGE = config.preprocess['BP_EDGE']
    lower_stop , upper_stop , lower_pass , upper_pass = BP_EDGE
    ws = [lower_stop/(fs*0.5), upper_stop/(fs*0.5)] # stopband edge frequencies
    wp = [lower_pass/(fs*0.5), upper_pass/(fs*0.5)] # passband edge frequencies
    rp = 3 # maximum loss in pb
    rs = 60 # minimum attenuation in sb
    N, Wn = signal.cheb2ord(wp,ws,rp,rs)
    iir_params = dict(ftype='cheby2', order=N, rp=rp, rs=rs, output='sos')
    iir_params = mne.filter.construct_iir_filter(iir_params, [lower_pass, upper_stop], sfreq=fs, btype='bandpass')  

    raw_cheb = data.copy().filter(lower_pass, upper_stop, iir_params=iir_params, method='iir');
    data_filt = raw_cheb.copy().notch_filter(50);
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
    ica_n_components = len(data.info['ch_names'])     # Specify n_components as a decimal to set % explained variance

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=ica_n_components,
                                random_state=random_state,
                                )
    ica.fit(epochs_ica,
            reject=reject,
            tstep=tstep)

    ica.plot_sources(inst=data, show=True)
    #ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': 40});

    exclude_channels = input("Input:")
    if len(exclude_channels) > 0:
        exclude_channels = [np.int64(v) for v in exclude_channels.split(",")]
    
    ica.exclude = exclude_channels
    data = ica.apply(data)

    return data, exclude_channels

def rename_channels(data):
    """
    rename channels to remove 'EEG', '-A1' and '-A2'
    """
    ch_names_dict = {}
    for i in range(data.get_data().shape[0]):
        ch_name = data.info['chs'][i]['ch_name']
        ch_name_new = ch_name
        ch_name_new = ch_name_new.replace('EEG ', '')
        ch_name_new = ch_name_new.replace('-A1', '')
        ch_name_new = ch_name_new.replace('-A2', '')
        ch_names_dict[ch_name] = ch_name_new

    mne.rename_channels(data.info, ch_names_dict)
    
    not_matching_channels = list(set(data.info['ch_names'][:-1]) - set(config['ch_names']))
    if len(not_matching_channels):
        data.info['bads'].extend(not_matching_channels)

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
