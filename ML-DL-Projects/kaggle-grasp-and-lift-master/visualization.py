#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np

import utils

from matplotlib.collections import LineCollection


# efficiently plot a multi-colored line
# http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'),
              norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm,
                        linewidth=linewidth, alpha=alpha)

    return lc


# convert events to colors, most recent events takes precedence
def get_colors(events):
    color_list = []
    for event in events.T:
        color_index = np.where(event == 1)[0]
        if color_index.shape[0] == 0:
            color_index = 0
        else:
            color_index = color_index[-1] + 1
        color_list.append(color_index / 6.)

    return color_list


# visualize the eight time series for a given subject, with the
# events highlighted in color
def visualize_subject(subj_id, sensor_id, num_points=10000):
    data_list, events_list = utils.load_subject_train(subj_id)
    print('visualizing time series for subject %d and sensor %d' %
          (subj_id, sensor_id))

    # 8 series organized in a 4 x 2 grid
    fig, axes = plt.subplots(4, 2, sharex=True)

    # hack to have a large figure when saving to disk
    fig.set_size_inches(15, 10)
    for series_id, (ax, data, events) in enumerate(zip(axes.flatten(),
                                                   data_list,
                                                   events_list),
                                                   start=1):
        # given sensors are indexed from 1
        signal = data[sensor_id - 1, :num_points]
        color_list = get_colors(events)

        x = np.arange(signal.shape[0])

        # change the cmap to have gray for the no-event signal
        cmap = plt.cm.nipy_spectral
        cmap_list = [cmap(i) for i in range(cmap.N)]
        cmap_list[0] = (0.5, 0.5, 0.5, 1.)
        cmap = cmap.from_list('Custom cmap', cmap_list, cmap.N)

        # plot the multi-colored line
        lc = colorline(x, signal,
                       z=color_list,
                       cmap=cmap)

        ax.add_collection(lc)
        ax.set_xlim((x.min(), x.max()))
        ax.set_ylim((signal.min(), signal.max()))
        ax.set_title('Series %d' % (series_id))

    plt.suptitle('Subject %d, Sensor %d' % (subj_id, sensor_id))
    #plt.show()
    out_file = 'subj%d_sensor%d_plot.png' % (subj_id, sensor_id)
    plt.savefig(out_file, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='Visualize the time'
                                     'series for a given subject and sensor')
    parser.add_argument('--subj', type=int, metavar='subj_id',
                        default=1,
                        choices=range(1, 13),
                        help='the id of the subject')
    parser.add_argument('--sensor', type=int, metavar='sensor_id',
                        default=1,
                        choices=range(1, 33),
                        help='the id of the sensor')
    parser.add_argument('--num', type=int, metavar='num_points',
                        default=10000,
                        help='the number of timesteps')

    args = parser.parse_args()

    subj_id = args.subj
    sensor_id = args.sensor
    num_points = args.num

    visualize_subject(subj_id, sensor_id, num_points=num_points)


if __name__ == '__main__':
    main()
