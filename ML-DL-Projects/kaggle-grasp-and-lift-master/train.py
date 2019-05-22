#!/usr/bin/env python

import cPickle as pickle
import numpy as np
import theano
import sys

from lasagne import layers
from os.path import join
from sklearn.metrics import roc_auc_score, log_loss
from time import time

import batching
import iter_funcs
import utils

#from convnet import build_model
#from convnet_small import build_model
#from convnet_deep import build_model
#from convnet_nopool import build_model
#from convnet_deep_drop import build_model
#from convnet_deep_drop_reg import build_model
#from convnet_bn import build_model
from convnet_csp import build_model
#from convnet_lr import build_model


def train_model(subj_id, window_size, max_epochs, patience):
    root_dir = join('data', 'nets')
    # the file from which to load pre-trained weights
    #init_file = join(root_dir,
    #                 'subj%d_weights.pickle' % (
    #                     subj_id))
    #init_file = join(root_dir,
    #                 'weights_deep.pickle')
    init_file = None
    # the file to which the learned weights will be written
    weights_file = join(root_dir,
                        'subj%d_csp_lr.pickle' % (
                            subj_id))
    print('loading time series for subject %d...' % (subj_id))
    data_list, events_list = utils.load_subject_train(subj_id)

    print('creating train and validation sets...')
    train_data, train_events, valid_data, valid_events = \
        utils.split_train_test_data(data_list, events_list,
                                    val_size=2, rand=False)
    print('using %d time series for training' % (len(train_data)))
    print('using %d time series for validation' % (len(valid_data)))

    train_data, valid_data = \
        utils.preprocess(train_data, valid_data)

    #train_data, train_events = utils.remove_easy_negatives(
    #    train_data, train_events)

    #valid_data, valid_events = utils.remove_easy_negatives(
    #    valid_data, valid_events)

    print('creating fixed-size time-windows of size %d' % (window_size))
    # the training windows should be in random order
    train_slices = batching.get_permuted_windows(train_data, window_size,
                                                 rand=True)
    valid_slices = batching.get_permuted_windows(valid_data, window_size,
                                                 rand=True)
    print('there are %d windows for training' % (len(train_slices)))
    print('there are %d windows for validation' % (len(valid_slices)))

    #train_slices = utils.duplicate_positives(train_slices, train_events, N=10)
    #print('there are %d windows for training' % (len(train_slices)))

    batch_size = 64
    #batch_size = 128
    #batch_size = 256
    #batch_size = 4096
    num_channels = 32
    num_actions = 6

    print('building model...')
    l_out = build_model(None, num_channels,
                        window_size, num_actions)
    print('building model %s...' % (
        sys.modules[build_model.__module__].__name__))
    print('  batch_size = %d' % (batch_size))
    print('  num_channels = %d' % (num_channels))
    print('  window_size = %d' % (window_size))

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

    #lr = theano.shared(np.cast['float32'](0.001))
    lr = theano.shared(np.cast['float32'](0.01))
    #lr = theano.shared(np.cast['float32'](0.1))
    mntm = 0.9
    print('compiling theano functions...')
    train_iter = iter_funcs.create_iter_funcs_train(lr, mntm, l_out)
    valid_iter = iter_funcs.create_iter_funcs_valid(l_out)

    best_weights = None
    best_valid_loss = np.inf
    best_epoch = 0
    print('starting training for subject %d at %s' % (
        subj_id, utils.get_current_time()))
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
                                        noisy=False)):
                t_batch_start = time()
                # hack for faster debugging
                #if i < 70000:
                #    continue
                train_loss, train_output = \
                    train_iter(Xb, yb)
                batch_duration = time() - t_batch_start
                if np.isnan(train_loss):
                    print('nan loss encountered in minibatch %d' % (i))
                    exit(0)
                if i % 100 == 0:
                    eta = batch_duration * (num_batches - i)
                    m, s = divmod(eta, 60)
                    h, m = divmod(m, 60)
                    print('  training...    (ETA = %d:%02d:%02d)\r'
                          % (h, m, s)),
                sys.stdout.flush()
                train_losses.append(train_loss)
                assert len(yb) == len(train_output)
                for input, output in zip(yb, train_output):
                    training_inputs.append(input)
                    training_outputs.append(output)
            avg_train_loss = np.mean(train_losses)

            training_inputs = np.vstack(training_inputs)
            training_outputs = np.vstack(training_outputs)
            train_roc = roc_auc_score(training_inputs, training_outputs)
            train_rocs = [roc_auc_score(
                training_inputs[:, i], training_outputs[:, i])
                for i in range(training_inputs.shape[1])]

            train_losses = [log_loss(
                training_inputs[:, i], training_outputs[:, i])
                for i in range(training_inputs.shape[1])]

            train_duration = time() - t_train_start
            print('')
            print('    train loss:  %.6f (%s)' % (
                avg_train_loss, ', '.join(['%.3f' % (loss)
                                          for loss in train_losses])))
            print('    train roc:   %.6f (%s)' % (
                train_roc, ', '.join(['%.3f' % (roc)
                                     for roc in train_rocs])))
            print('    duration:    %.2f s' % (train_duration))

            valid_losses, valid_outputs, valid_inputs = [], [], []
            num_batches = (len(valid_slices) + batch_size - 1) / batch_size
            t_valid_start = time()
            for i, (Xb, yb) in enumerate(
                batching.batch_iterator(batch_size,
                                        valid_slices,
                                        valid_data,
                                        valid_events,
                                        noisy=False)):
                #augmented_valid_losses, augmented_valid_outputs = [], []
                #for offset in range(0, subsample):
                #    valid_loss, valid_output = \
                #        valid_iter(Xb[:, :, offset::subsample], yb)
                #    augmented_valid_losses.append(valid_loss)
                #    augmented_valid_outputs.append(valid_output)
                #valid_loss = np.mean(augmented_valid_losses)
                #valid_output = batching.compute_geometric_mean(
                #    augmented_valid_outputs)
                t_batch_start = time()
                valid_loss, valid_output = \
                    valid_iter(Xb, yb)
                if np.isnan(valid_loss):
                    print('nan loss encountered in minibatch %d' % (i))
                    continue
                batch_duration = time() - t_batch_start
                if i % 10 == 0:
                    eta = batch_duration * (num_batches - i)
                    m, s = divmod(eta, 60)
                    h, m = divmod(m, 60)
                    print('  validation...  (ETA = %d:%02d:%02d)\r'
                          % (h, m, s)),
                    sys.stdout.flush()

                valid_losses.append(valid_loss)
                assert len(yb) == len(valid_output)
                for input, output in zip(yb, valid_output):
                    valid_inputs.append(input)
                    valid_outputs.append(output)

            # allow training without validation
            if valid_losses:
                avg_valid_loss = np.mean(valid_losses)
                valid_inputs = np.vstack(valid_inputs)
                valid_outputs = np.vstack(valid_outputs)
                valid_roc = roc_auc_score(valid_inputs, valid_outputs)
                valid_rocs = [roc_auc_score(
                    valid_inputs[:, i], valid_outputs[:, i])
                    for i in range(valid_inputs.shape[1])]

                valid_losses = [log_loss(
                    valid_inputs[:, i], valid_outputs[:, i])
                    for i in range(valid_inputs.shape[1])]

                valid_duration = time() - t_valid_start
                print('')
                print('    valid loss:  %.6f (%s)' % (
                    avg_valid_loss, ', '.join(['%.3f' % (loss)
                                              for loss in valid_losses])))
                print('    valid roc:   %.6f (%s)' % (
                    valid_roc, ', '.join(['%.3f' % (roc)
                                         for roc in valid_rocs])))
                print('    duration:    %.2f s' % (valid_duration))
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
    print('finished training for subject %d at %s' % (
        subj_id, utils.get_current_time()))

    return model_train_loss, best_valid_loss, model_train_roc, model_valid_roc


def main():
    #subjects = range(5, 13)
    #subjects = range(6, 13)
    subjects = [5, 8, 12]
    window_size = 2000
    #window_size = 200
    #window_size = 1000
    max_epochs = 5
    patience = 1
    #max_epochs = 5
    model_train_losses, model_valid_losses = [], []
    model_train_rocs, model_valid_rocs = [], []
    for subj_id in subjects:
        model_train_loss, model_valid_loss, model_train_roc, model_valid_roc =\
            train_model(subj_id, window_size, max_epochs, patience)
        print('\n%s subject %d %s' % ('*' * 10, subj_id, '*' * 10))
        print(' model training loss = %.5f' % (model_train_loss))
        print(' model valid loss    = %.5f' % (model_valid_loss))
        print(' model training roc  = %.5f' % (model_train_roc))
        print(' model valid roc     = %.5f' % (model_valid_roc))
        print('%s subject %d %s\n' % ('*' * 10, subj_id, '*' * 10))
        model_train_losses.append(model_train_loss)
        model_valid_losses.append(model_valid_loss)
        model_train_rocs.append(model_train_roc)
        model_valid_rocs.append(model_valid_roc)

    print('average loss over subjects {%s}:' %
          (' '.join([str(s) for s in subjects])))
    print('  training loss:   %.5f' % (np.mean(model_train_losses)))
    print('  validation loss: %.5f' % (np.mean(model_valid_losses)))
    print('  training roc:    %.5f' % (np.mean(model_train_rocs)))
    print('  validation roc:  %.5f' % (np.mean(model_valid_rocs)))


if __name__ == '__main__':
    main()
