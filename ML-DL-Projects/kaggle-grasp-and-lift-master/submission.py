#!/usr/bin/env python

import cPickle as pickle
import numpy as np
from lasagne import layers
from os.path import join

import batching
import iter_funcs
import utils

#from convnet import build_model
#from convnet_small import build_model
#from convnet_deep import build_model
from convnet_deep_drop import build_model
#from convnet_regions import build_model


def generate_submission(subj_id, window_size):
    weights_dir = join('data', 'nets')

    batch_size = 256
    num_channels = 32
    num_actions = 6
    print('building model...')
    l_out = build_model(None, num_channels,
                        window_size, num_actions)

    print('predicting for subj_id %d...' % (subj_id))
    preds_file = join('data', 'predictions',
                      'subj%d_preds_super.csv' %
                      subj_id)
    weights_file = join(weights_dir,
                        'subj%d_weights_deep_nocsp_wide.pickle' %
                        subj_id)

    print('loading model weights from %s' % (weights_file))
    with open(weights_file, 'rb') as ifile:
        model_params = pickle.load(ifile)
    layers.set_all_param_values(l_out, model_params)

    print('loading time series for subject %d...' % (subj_id))
    data_list, events_list = utils.load_subject_train(subj_id)
    test_data, test_ids = utils.load_subject_test(subj_id)

    print('creating train and validation sets...')
    train_data, train_events, valid_data, valid_events = \
        utils.split_train_test_data(data_list, events_list,
                                    val_size=2, rand=False)

    test_data, test_ids = utils.load_subject_test(subj_id)

    print('compiling theano functions...')
    test_iter = iter_funcs.create_iter_funcs_test(l_out)

    print('getting time windows for test data')
    # the test windows should be in fixed order
    test_slices = batching.get_permuted_windows(test_data, window_size,
                                                rand=False)
    print('there are %d windows for prediction' % (len(test_slices)))

    print('preprocessing...')
    train_data, test_data = \
        utils.preprocess(subj_id, train_data, test_data)

    for data in test_data:
        print('data.shape = %r' % (data.shape,))

    test_outputs = []
    print('predicting...')
    for i, (Xb, _) in enumerate(batching.batch_iterator(batch_size,
                                test_slices,
                                test_data,
                                y=None,
                                window_norm=False)):
        test_output = test_iter(Xb)
        for output in test_output:
            test_outputs.append(output)

    print('writing %d output probabilities to %s...' % (
        len(test_outputs), preds_file))
    with open(preds_file, 'w') as ofile:
        output_index = 0
        for series_id, series_test_ids in enumerate(test_ids):
            for i, test_id in enumerate(series_test_ids):
                if i < window_size:
                    zeros = np.zeros(6, dtype=np.float32)
                    ofile.write('%s,%s\n' % (
                        test_id, ','.join(['%.3f' % z for z in zeros])))
                else:
                    probs = test_outputs[output_index]
                    output_index += 1
                    ofile.write('%s,%s\n' % (
                        test_id, ','.join(['%.3f' % p for p in probs])))


def main():
    #subjects = range(1, 6)
    subjects = [1, 2, 7, 8, 9, 10, 11, 12]
    #subjects = [6, 7, 8, 10]
    #subjects = range(6, 13)
    window_size = 2000

    for subj_id in subjects:
        generate_submission(subj_id, window_size)


if __name__ == '__main__':
    main()
