#!/usr/bin/env python

import cPickle as pickle
import numpy as np
import theano
import sys

from lasagne import layers
from os.path import join
from sklearn.metrics import roc_auc_score
from time import time

import batching
import iter_funcs
import utils

from convnet_deep_drop import build_model


def train_model(window_size, max_epochs, patience):
    root_dir = join('data', 'nets')
    # the file from which to load pre-trained weights
    #init_file = join(root_dir,
    #                 'subj%d_weights_deep_nocsp_wide.pickle' % (
    #                     4))
    #init_file = join(root_dir,
    #                 'weights_super_deeper.pickle')
    init_file = None
    # the file to which the learned weights will be written
    weights_file = join(root_dir,
                        'weights.pickle')
    temp_weights_file = join(root_dir, 'epoch_%d.pickle')
    train_data, train_events = [], []
    valid_data, valid_events = [], []
    for subj_id in range(1, 13):
        print('loading time series for subject %d...' % (subj_id))
        subj_data_list, subj_events_list = utils.load_subject_train(subj_id)
        print('  creating train and validation sets...')
        subj_train_data, subj_train_events, subj_valid_data, subj_valid_events = \
            utils.split_train_test_data(subj_data_list, subj_events_list,
                                        val_size=2, rand=False)
        train_data += subj_train_data
        train_events += subj_train_events
        valid_data += subj_valid_data
        valid_events += subj_valid_events

    print('using %d time series for training' % (len(train_data)))
    print('using %d time series for validation' % (len(valid_data)))

    print('creating fixed-size time-windows of size %d' % (window_size))
    # the training windows should be in random order
    train_slices = batching.get_permuted_windows(train_data, window_size,
                                                 rand=True)
    valid_slices = batching.get_permuted_windows(valid_data, window_size,
                                                 rand=True)
    print('there are %d windows for training' % (len(train_slices)))
    print('there are %d windows for validation' % (len(valid_slices)))

    #batch_size = 64
    batch_size = 512
    num_channels = 32
    num_actions = 6
    train_data, valid_data = \
        utils.preprocess(train_data, valid_data)

    print('building model %s...' % (
        sys.modules[build_model.__module__].__name__))
    l_out = build_model(None, num_channels,
                        window_size, num_actions)

    all_layers = layers.get_all_layers(l_out)
    print('this network has %d learnable parameters' %
          (layers.count_params(l_out)))
    for layer in all_layers:
        print('Layer %s has output shape %r' %
              (layer.name, layer.output_shape))

    if init_file is not None:
        print('loading model weights from %s' % (init_file))
        with open(init_file, 'rb') as ifile:
            src_layers = pickle.load(ifile)
        dst_layers = layers.get_all_params(l_out)
        for i, (src_weights, dst_layer) in enumerate(
                zip(src_layers, dst_layers)):
            print('loading pretrained weights for %s' % (dst_layer.name))
            dst_layer.set_value(src_weights)
    else:
        print('all layers will be trained from random initialization')

    #1r = theano.shared(np.cast['float32'](0.001))
    lr = theano.shared(np.cast['float32'](0.01))
    mntm = 0.9
    print('compiling theano functions...')
    train_iter = iter_funcs.create_iter_funcs_train(lr, mntm, l_out)
    valid_iter = iter_funcs.create_iter_funcs_valid(l_out)

    best_weights = None
    best_valid_loss = np.inf
    best_epoch = 0
    print('starting training for all subjects at %s' % (
        utils.get_current_time()))
    try:
        for epoch in range(max_epochs):
            print('epoch: %d' % (epoch))
            train_losses, training_outputs, training_inputs = [], [], []
            num_batches = (len(train_slices) + batch_size - 1) / batch_size
            t_train_start = time()
            for i, (Xb, yb) in enumerate(
                batching.batch_iterator(batch_size,
                                        train_slices,
                                        train_data,
                                        train_events,
                                        window_norm=False)):
                t_batch_start = time()
                # hack for faster debugging
                #if i < 70000:
                #    continue
                train_loss, train_output = \
                    train_iter(Xb, yb)
                if np.isnan(train_loss):
                    print('nan loss encountered in minibatch %d' % (i))
                    continue

                train_losses.append(train_loss)
                assert len(yb) == len(train_output)
                for input, output in zip(yb, train_output):
                    training_inputs.append(input)
                    training_outputs.append(output)

                batch_duration = time() - t_batch_start
                if i % 10 == 0:
                    eta = batch_duration * (num_batches - i)
                    m, s = divmod(eta, 60)
                    h, m = divmod(m, 60)
                    print('  training...  (ETA = %d:%02d:%02d)\r'
                          % (h, m, s)),
                    sys.stdout.flush()

            avg_train_loss = np.mean(train_losses)

            training_inputs = np.vstack(training_inputs)
            training_outputs = np.vstack(training_outputs)
            train_roc = roc_auc_score(training_inputs, training_outputs)

            train_duration = time() - t_train_start
            print('')
            print('    train loss: %.6f' % (avg_train_loss))
            print('    train roc:  %.6f' % (train_roc))
            print('    duration:   %.2f s' % (train_duration))

            valid_losses, valid_outputs, valid_inputs = [], [], []
            num_batches = (len(valid_slices) + batch_size - 1) / batch_size
            t_valid_start = time()
            for i, (Xb, yb) in enumerate(
                batching.batch_iterator(batch_size,
                                        valid_slices,
                                        valid_data,
                                        valid_events,
                                        window_norm=False)):
                t_batch_start = time()
                valid_loss, valid_output = \
                    valid_iter(Xb, yb)
                if np.isnan(valid_loss):
                    print('nan loss encountered in minibatch %d' % (i))
                    continue
                valid_losses.append(valid_loss)
                assert len(yb) == len(valid_output)
                for input, output in zip(yb, valid_output):
                    valid_inputs.append(input)
                    valid_outputs.append(output)

                batch_duration = time() - t_batch_start
                if i % 10 == 0:
                    eta = batch_duration * (num_batches - i)
                    m, s = divmod(eta, 60)
                    h, m = divmod(m, 60)
                    print('  validation...  (ETA = %d:%02d:%02d)\r'
                          % (h, m, s)),
                    sys.stdout.flush()

            # allow training without validation
            if valid_losses:
                avg_valid_loss = np.mean(valid_losses)
                valid_inputs = np.vstack(valid_inputs)
                valid_outputs = np.vstack(valid_outputs)
                valid_roc = roc_auc_score(valid_inputs, valid_outputs)
                valid_duration = time() - t_valid_start
                print('')
                print('    valid loss: %.6f' % (avg_valid_loss))
                print('    valid roc:  %.6f' % (valid_roc))
                print('    duration:   %.2f s' % (valid_duration))
            else:
                print('    no validation...')

            # if we are not doing validation we always want the latest weights
            if not valid_losses:
                best_epoch = epoch
                model_train_loss = avg_train_loss
                model_train_roc = train_roc
                model_valid_roc = -1.
                best_valid_loss = -1.
                best_weights = layers.get_all_param_values(l_out)
            elif avg_valid_loss < best_valid_loss:
                best_epoch = epoch
                model_train_roc = train_roc
                model_valid_roc = valid_roc
                model_train_loss = avg_train_loss
                best_valid_loss = avg_valid_loss
                best_weights = layers.get_all_param_values(l_out)

                temp_file = temp_weights_file % (epoch)
                print('saving temporary best weights to %s' % (temp_file))
                with open(temp_file, 'wb') as ofile:
                    pickle.dump(best_weights, ofile,
                                protocol=pickle.HIGHEST_PROTOCOL)

            if epoch > best_epoch + patience:
                break
                best_epoch = epoch
                new_lr = 0.5 * lr.get_value()
                lr.set_value(np.cast['float32'](new_lr))
                print('setting learning rate to %.6f' % (new_lr))

    except KeyboardInterrupt:
        print('caught Ctrl-C, stopping training...')

    with open(weights_file, 'wb') as ofile:
        print('saving best weights to %s' % (weights_file))
        pickle.dump(best_weights, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    print('finished training for all subjects at %s' % (
        utils.get_current_time()))

    return model_train_loss, best_valid_loss, model_train_roc, model_valid_roc


def main():
    window_size = 2000
    #window_size = 1000
    max_epochs = 10
    patience = 1
    model_train_loss, model_valid_loss, model_train_roc, model_valid_roc =\
        train_model(window_size, max_epochs, patience)
    print('\n%s all subjects %s' % ('*' * 10, '*' * 10))
    print(' model training loss = %.5f' % (model_train_loss))
    print(' model valid loss    = %.5f' % (model_valid_loss))
    print(' model training roc  = %.5f' % (model_train_roc))
    print(' model valid roc     = %.5f' % (model_valid_roc))
    print('%s all subjects %s\n' % ('*' * 10, '*' * 10))


if __name__ == '__main__':
    main()
