#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import utils
import visualization


def visualize_subject(subj_id, num_points=10000):
    start_offset = 2000
    print('loading time series for subject %d...' % (subj_id))
    data_list, events_list = utils.load_subject_train(subj_id)
    test_data, test_ids = utils.load_subject_test(subj_id)

    print('creating train and validation sets...')
    train_data, train_events, valid_data, valid_events = \
        utils.split_train_test_data(data_list, events_list,
                                    val_size=2, rand=False)
    print('using %d time series for training' % (len(train_data)))
    print('using %d time series for validation' % (len(valid_data)))
    original_train_data = train_data[:]
    print('preprocessing validation data...')
    train_data, valid_data = \
        utils.preprocess(subj_id, train_data, valid_data, compute_csp=True)
    print('preprocessing test data...')
    train_data, test_data = \
        utils.preprocess(subj_id, original_train_data, test_data)

    fig, axes = plt.subplots(10, 4, sharex=True)
    fig.set_size_inches(20, 15)

    # change the cmap to have gray for the no-event signal
    cmap = plt.cm.nipy_spectral
    cmap_list = [cmap(i) for i in range(cmap.N)]
    cmap_list[0] = (0.5, 0.5, 0.5, 1.)
    cmap = cmap.from_list('Custom cmap', cmap_list, cmap.N)

    for series_id in range(0, 8):
        print('plotting train time-series %d...' % (series_id + 1))
        for channel_id in range(0, 4):
            ax = axes[series_id, channel_id]
            if series_id < 6:
                signal = train_data[series_id][channel_id][
                    start_offset:start_offset + num_points]
                events = train_events[series_id]
            else:
                signal = valid_data[series_id - 6][channel_id][
                    start_offset:start_offset + num_points]
                events = valid_events[series_id - 6]

            color_list = visualization.get_colors(events)

            x = np.arange(signal.shape[0])

            # plot the multi-colored line
            lc = visualization.colorline(x, signal,
                                         z=color_list,
                                         cmap=cmap)

            ax.add_collection(lc)
            ax.set_xlim((x.min(), x.max()))
            ax.set_ylim((signal.min(), signal.max()))
            ax.set_title('Channel %d' % (channel_id))

    for series_id in range(8, 10):
        print('plotting train time-series %d...' % (series_id + 1))
        for channel_id in range(0, 4):
            ax = axes[series_id, channel_id]
            signal = test_data[9 - series_id][channel_id][
                start_offset:start_offset + num_points]
            x = np.arange(signal.shape[0])
            ax.plot(x, signal, color=cmap_list[0])
            ax.set_xlim((x.min(), x.max()))
            ax.set_ylim((signal.min(), signal.max()))
            ax.set_title('Channel %d' % (channel_id))

    plt.suptitle('Subject %d' % (subj_id))
    out_file = 'subj%d_plot.png' % (subj_id)
    plt.savefig(out_file, bbox_inches='tight')

if __name__ == '__main__':
    visualize_subject(1, num_points=10000)
