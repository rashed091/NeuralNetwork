#!/usr/bin/env python

import batching
import iter_funcs
import utils

import cPickle as pickle
import numpy as np
from convnet_deep_drop import build_model

from lasagne import layers
from os.path import join


def generate_submission(subjects, window_size):
    weights_dir = join('data', 'nets')
    weights_file = join(weights_dir,
                        'weights_super_fixed.pickle')
    preds_file_template = join('data', 'predictions',
                               'subj%d_fixed.csv')

    train_data, train_events = [], []
    # need to normalize over all subjects even if we're not predicting for all
    for subj_id in range(1, 13):
        print('loading training time series for subject %d...' % (subj_id))
        subj_data_list, subj_events_list = utils.load_subject_train(subj_id)
        print('  creating train and validation sets...')
        subj_train_data, subj_train_events, subj_valid_data, subj_valid_events = \
            utils.split_train_test_data(subj_data_list, subj_events_list,
                                        val_size=2, rand=False)

        train_data += subj_train_data
        train_events += subj_train_events

    batch_size = 512
    num_channels = 32
    num_actions = 6

    print('building model...')
    l_out = build_model(None, num_channels,
                        window_size, num_actions)

    print('loading model weights from %s' % (weights_file))
    with open(weights_file, 'rb') as ifile:
        model_params = pickle.load(ifile)
    layers.set_all_param_values(l_out, model_params)

    print('compiling theano functions...')
    test_iter = iter_funcs.create_iter_funcs_test(l_out)

    for subj_id in subjects:
        print('loading test time series for subject %d...' % (subj_id))
        test_data, test_ids = utils.load_subject_test(subj_id)
        print('preprocessing...')
        train_data, test_data = \
            utils.preprocess(train_data, test_data)

        test_data = batching.pad_test_series(test_data, window_size)

        # the test windows should be in fixed order
        test_slices = batching.get_permuted_windows(test_data, window_size,
                                                    rand=False)
        print('predicting for %d windows of subject %d...' % (
            len(test_slices), subj_id))
        test_outputs = []
        for i, (Xb, _) in enumerate(batching.batch_iterator(batch_size,
                                    test_slices,
                                    test_data,
                                    y=None,
                                    window_norm=False)):
            test_output = test_iter(Xb)
            for output in test_output:
                test_outputs.append(output)

        preds_file = preds_file_template % (subj_id)
        print('writing %d output probabilities to %s...' % (
            len(test_outputs), preds_file))
        with open(preds_file, 'w') as ofile:
            output_index = 0
            for series_id, series_subj_ids in enumerate(test_ids):
                for i, test_id in enumerate(series_subj_ids):
                    #if i < window_size:
                    #    zeros = np.zeros(6, dtype=np.float32)
                    #    ofile.write('%s,%s\n' % (
                    #        test_id, ','.join(['%.3f' % z for z in zeros])))
                    #else:
                    #    probs = test_outputs[output_index]
                    #    output_index += 1
                    #    ofile.write('%s,%s\n' % (
                    #        test_id, ','.join(['%.3f' % p for p in probs])))
                    probs = test_outputs[output_index]
                    output_index += 1
                    ofile.write('%s,%s\n' % (
                        test_id, ','.join(['%.3f' % p for p in probs])))


if __name__ == '__main__':
    subjects = range(1, 13)
    window_size = 2000
    generate_submission(subjects, window_size)
