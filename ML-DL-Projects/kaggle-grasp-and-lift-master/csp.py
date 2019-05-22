import numpy as np
import pandas as pd

from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP  # NOQA
from glob import glob
from scipy.signal import butter, lfilter, convolve, boxcar


def create_mne_raw_object(fname, read_events=True):
    """Create a mne raw instance from csv file"""
    # Read EEG file
    data = pd.read_csv(fname)

    # get chanel names
    ch_names = list(data.columns[1:])

    # read EEG standard montage from mne
    montage = read_montage('standard_1005', ch_names)

    ch_type = ['eeg'] * len(ch_names)
    data = 1e-6 * np.array(data[ch_names]).T

    if read_events:
        # events file
        ev_fname = fname.replace('_data', '_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T

        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim'] * 6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data, events_data))

    # create and populate MNE info structure
    info = create_info(ch_names, sfreq=500.0,
                       ch_types=ch_type, montage=montage)
    info['filename'] = fname

    # create raw object
    raw = RawArray(data, info, verbose=False)

    return raw


def compute_transform(subj_id, nfilters=4):
    freqs = [7, 30]
    b, a = butter(5, np.array(freqs) / 250.0, btype='bandpass')
    epochs_tot = []
    y = []
    fnames =  glob('data/train/subj%d_series*_data.csv' % (subj_id))

    train_raw = concatenate_raws([create_mne_raw_object(fname)
                                  for fname in fnames])

    picks = pick_types(train_raw.info, eeg=True)

    train_raw._data[picks] = lfilter(b, a, train_raw._data[picks])

    events = find_events(train_raw, stim_channel='Replace', verbose=False)
    epochs = Epochs(train_raw, events, {'during' : 1}, -2, -0.5, proj=False,
                    picks=picks, baseline=None, preload=True,
                    add_eeg_ref=False, verbose=False)

    epochs_tot.append(epochs)
    y.extend([1] * len(epochs))

    epochs_rest = Epochs(train_raw, events, {'after' : 1}, 0.5, 2, proj=False,
                         picks=picks, baseline=None, preload=True,
                         add_eeg_ref=False, verbose=False)

    epochs_rest.times = epochs.times

    y.extend([-1] * len(epochs_rest))
    epochs_tot.append(epochs_rest)

    # Concatenate all epochs
    epochs = concatenate_epochs(epochs_tot)

    # get data
    X = epochs.get_data()
    y = np.array(y)

    # train CSP
    csp = CSP(n_components=nfilters, reg='lws')
    csp.fit(X, y)

    #from pyriemann.spatialfilters import Xdawn
    #xdawn = Xdawn(nfilter=nfilters / 2)
    #xdawn.fit(X, y)

    #return xdawn.V
    return csp.filters_[:nfilters]


def apply_transform(series_list, transform):
    series_list_out = []
    for series in series_list:
        transformed = np.dot(transform, series)
        series_list_out.append(transformed)

    return series_list_out


def post_csp(series_list, nwin=250):
    series_list_out = []
    series_list = [series ** 2 for series in series_list]
    for series in series_list:
        nfilters = series.shape[0]
        series_out = np.empty(series.shape)
        for i in range(nfilters):
            series_out[i] = np.log(convolve(series[i],
                                            boxcar(nwin), 'full')
                                   )[0:series.shape[1]]
        series_list_out.append(series_out)

    return series_list_out
