from __future__ import division, print_function, absolute_import
import os, sys
from collections import OrderedDict
import numpy as np

import tensorflow as tf

from tfutils import base, data, model, optimizer, utils

import copy

# toggle this to train or to validate at the end
train_net = True
# toggle this to train on whitenoise or naturalscene data
stim_type = 'whitenoise'
#stim_type = 'naturalscene'
# Figure out the hostname
host = os.uname()[1]
if 'neuroaicluster' in host:
    if train_net:
        print('In train mode...')
        TOTAL_BATCH_SIZE = 5000
        MB_SIZE = 5000
        NUM_GPUS = 1
    else:
        print('In val mode...')
        if stim_type == 'whitenoise':
            TOTAL_BATCH_SIZE = 5957
            MB_SIZE = 5957
            NUM_GPUS = 1
        else:
            TOTAL_BATCH_SIZE = 5956
            MB_SIZE = 5956
            NUM_GPUS = 1

else:
    print("Data path not found!!")
    exit()

if not isinstance(NUM_GPUS, list):
    DEVICES = ['/gpu:' + str(i) for i in range(NUM_GPUS)]
else:
    DEVICES = ['/gpu:' + str(i) for i in range(len(NUM_GPUS))]

MODEL_PREFIX = 'model_0'

# Data parameters
if stim_type == 'whitenoise':
    N_TRAIN = 323762
    N_TEST = 5957
else:
    N_TRAIN = 323756
    N_TEST = 5956

INPUT_BATCH_SIZE = 1024 # queue size
OUTPUT_BATCH_SIZE = TOTAL_BATCH_SIZE
print('TOTAL BATCH SIZE:', OUTPUT_BATCH_SIZE)
NUM_BATCHES_PER_EPOCH = N_TRAIN // OUTPUT_BATCH_SIZE
IMAGE_SIZE_RESIZE = 50

DATA_PATH = '/datasets/deepretina_data/tf_records/' + stim_type
print('Data path: ', DATA_PATH)

# data provider
class retinaTF(data.TFRecordsParallelByFileProvider):

  def __init__(self,
               source_dirs,
               resize=IMAGE_SIZE_RESIZE,
               **kwargs
               ):

    if resize is None:
      self.resize = 50
    else:
      self.resize = resize

    postprocess = {'images': [], 'labels': []}
    postprocess['images'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
    postprocess['images'].insert(1, (tf.reshape, ([-1] + [50, 50, 40], ), {}))
    postprocess['images'].insert(2, (self.postproc_imgs, (), {})) 

    postprocess['labels'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
    postprocess['labels'].insert(1, (tf.reshape, ([-1] + [5], ), {}))

    super(retinaTF, self).__init__(
      source_dirs,
      postprocess=postprocess,
      **kwargs
    )


  def postproc_imgs(self, ims):
    def _postprocess_images(im):
        im = tf.image.resize_images(im, [self.resize, self.resize])
        return im
    return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)

def ln(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, seed=0, cfg_final=None):
    params = OrderedDict()
    batch_size = inputs['images'].get_shape().as_list()[0]
    params['stim_type'] = stim_type
    params['train'] = train
    params['batch_size'] = batch_size

    # implement your LN model here

    return out, params

def cnn(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, seed=0, cfg_final=None):
    params = OrderedDict()
    batch_size = inputs['images'].get_shape().as_list()[0]
    params['stim_type'] = stim_type
    params['train'] = train
    params['batch_size'] = batch_size

    # implement your CNN here

    return out, params

def poisson_loss(logits, labels):
    # implement the poisson loss here

def mean_loss_with_reg(loss):
    return tf.reduce_mean(loss) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

def loss_metric(inputs, outputs, target, **kwargs):
    metrics_dict = {}
    metrics_dict['poisson_loss'] = mean_loss_with_reg(poisson_loss(logits=outputs, labels=inputs[target]), **kwargs)
    return metrics_dict

def mean_losses_keep_rest(step_results):
    retval = {}
    keys = step_results[0].keys()
    print('KEYS: ', keys)
    for k in keys:
        plucked = [d[k] for d in step_results]
        if isinstance(k, str) and 'loss' in k:
            retval[k] = np.mean(plucked)
        else:
            retval[k] = plucked
    return retval

# model parameters

default_params = {
    'save_params': {
        'host': 'localhost',
        'port': 27017,
        'dbname': 'deepretina',
        'collname': stim_type,
        'exp_id': 'trainval0',

        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': 50,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_valid_freq': 50,
        'save_filters_freq': 50,
        'cache_filters_freq': 50,
        # 'cache_dir': None,  # defaults to '~/.tfutils'
    },

    'load_params': {
        'do_restore': False,
        'query': None
    },

    'model_params': {
        'func': ln,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    },

    'train_params': {
        'minibatch_size': MB_SIZE,
        'data_params': {
            'func': retinaTF,
            'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
            'resize': IMAGE_SIZE_RESIZE,
            'batch_size': INPUT_BATCH_SIZE,
            'file_pattern': 'train*.tfrecords',
            'n_threads': 4
        },
        'queue_params': {
            'queue_type': 'random',
            'batch_size': OUTPUT_BATCH_SIZE,
            'capacity': 11*INPUT_BATCH_SIZE,
            'min_after_dequeue': 10*INPUT_BATCH_SIZE,
            'seed': 0,
        },
        'thres_loss': float('inf'),
        'num_steps': 50 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'validate_first': True,
    },

    'loss_params': {
        'targets': ['labels'],
        'agg_func': mean_loss_with_reg,
        'loss_per_case_func': poisson_loss
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 1e-3,
        'decay_rate': 1.0, # constant learning rate
        'decay_steps': NUM_BATCHES_PER_EPOCH,
        'staircase': True
    },

    'optimizer_params': {
        'func': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.AdamOptimizer,
        'clip': True,
        'trainable_names': None
    },

    'validation_params': {
        'test_loss': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'test*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': loss_metric,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': N_TEST // MB_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        },
        'train_loss': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'train*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': loss_metric,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': N_TRAIN // OUTPUT_BATCH_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        }

    },
    'log_device_placement': False,  # if variable placement has to be logged
}

def train_ln():
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'ln_model'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    params['model_params'] = # FILL IN HERE
    params['learning_rate_params']['learning_rate'] = 1e-3
    base.train_from_params(**params)

def train_cnn():
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'cnn'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    params['model_params'] = # FILL IN HERE
    params['learning_rate_params']['learning_rate'] = 1e-3
    base.train_from_params(**params)
 
if __name__ == '__main__':
    # train_cnn
    # train_ln


