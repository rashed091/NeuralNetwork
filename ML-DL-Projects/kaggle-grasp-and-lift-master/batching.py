#!/usr/bin/env python

import numpy as np
import random


def pad_test_series(test_data, window_size):
    test_data_padded = []
    for data in test_data:
        padding = np.zeros((data.shape[0], window_size - 1), np.float32)
        data = np.hstack((padding, data))
        test_data_padded.append(data)

    return test_data_padded


# return slices that chunk a time-series into windows
# of size window_size
def get_series_window_slices(num_datapoints, window_size):
    slices = []
    num_windows = num_datapoints - window_size + 1
    for i in range(0, num_windows, 1):
        slices.append(slice(i, i + window_size))

    return slices


# permute the time-windows of all time-series to
# give a shuffled list of time-windows over all
# time series
def get_permuted_windows(series_list, window_size, rand=True):
    series_slices = []
    for i, series in enumerate(series_list):
        slices = get_series_window_slices(series.shape[1],
                                          window_size)
        # need to mark each slice with the series it came from
        for s in slices:
            series_slices.append((i, s))

    # wish to iterate over the windows in random order for training
    if rand:
        random.shuffle(series_slices)
    return series_slices


def normalize_window(X):
    # normalize each channel's signal to be between 0 and 1
    X_min = np.min(X, axis=1).reshape(-1, 1)
    X_max = np.max(X, axis=1).reshape(-1, 1)

    return (X - X_min) / (X_max - X_min)


# splits the list of window indices and slices into batches
# and grabs the fixed-length windows from the corresponding
# slice from that time-series
def batch_iterator(bs, W, X, y=None, noisy=False):
    if not W:
        raise StopIteration
    window_size = W[0][1].stop - W[0][1].start
    # total number of batches for this data set and batch size
    N = (len(W) + bs - 1) / bs
    for i in range(N):
        Wb = W[i * bs:(i + 1) * bs]

        X_batch_list, y_batch_list = [], []
        # index: which time series to take the window from
        # s:     the slice to take from that time series
        for j, (index, s) in enumerate(Wb):
            X_window = X[index][:, s]
            if y is not None:
                y_window = y[index][:, s][:, -1]
                y_batch_list.append(y_window)

            # this is test data, train data but no action is present,
            # or validation data (don't want noise)
            if y is None or y_window.sum() == 0 or not noisy:
                X_batch_list.append(X_window)
            # this is train data and an action is present, so add noise
            else:
                noise = np.random.normal(
                    0, 0.1, X_window.shape).astype(np.float32)
                X_batch_list.append(X_window + noise)

        # reshape to (batch_size, num_channels, window_size)
        X_batch = np.vstack(X_batch_list).reshape(-1,
                                                  X[0].shape[0], window_size)
        if not y_batch_list:
            y_batch = None
        else:
            y_batch = np.vstack(y_batch_list)

        yield X_batch, y_batch


if __name__ == '__main__':
    #data = np.arange(5 * 10).reshape(5, 10)
    #slices = get_series_window_slices(data.shape[1], 3)
    #print data
    #print slices
    #print('slicing:')
    #for s in slices:
    #    print data[:, s]

    from sklearn.metrics import log_loss, roc_auc_score
    p1 = np.random.uniform(0, 1, (2000, 6))
    p2 = np.random.uniform(0, 1, (2000, 6))
    p3 = np.random.uniform(0, 1, (2000, 6))

    labels = np.random.randint(0, 2, (2000, 6))

    avg = compute_geometric_mean([p1, p2, p3])
    loss = log_loss(labels, avg)
    roc = roc_auc_score(labels, avg)

    print('loss = %.5f' % loss)
    print('roc = %.5f' % roc)
