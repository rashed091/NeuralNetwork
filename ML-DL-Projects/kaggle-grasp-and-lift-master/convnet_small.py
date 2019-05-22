from lasagne import layers
from lasagne import nonlinearities
from lasagne import init
#from lasagne.layers import cuda_convnet
from lasagne.layers import conv
from lasagne.layers import pool

#Conv1DLayer = cuda_convnet.Conv1DCCLayer
#MaxPool1DLayer = cuda_convnet.MaxPool1DCCLayer

Conv1DLayer = conv.Conv1DLayer
MaxPool1DLayer = pool.MaxPool1DLayer


def build_model(batch_size,
                num_channels,
                input_length,
                output_dim,):
    l_in = layers.InputLayer(
        shape=(batch_size, num_channels, input_length),
        name='l_in',
    )

    l_conv1 = Conv1DLayer(
        l_in,
        name='conv1',
        num_filters=8,
        border_mode='valid',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_pool1 = MaxPool1DLayer(
        l_conv1,
        name='pool1',
        pool_size=3,
        stride=2,
    )

    l_conv2 = Conv1DLayer(
        l_pool1,
        name='conv2',
        num_filters=16,
        border_mode='valid',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_pool2 = MaxPool1DLayer(
        l_conv2,
        name='pool2',
        pool_size=3,
        stride=2,
    )

    l_dropout_dense1 = layers.DropoutLayer(
        #l_pool4,
        l_pool2,
        p=0.5,
    )

    l_dense1 = layers.DenseLayer(
        l_dropout_dense1,
        num_units=32,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_out = layers.DenseLayer(
        l_dense1,
        num_units=output_dim,
        nonlinearity=nonlinearities.sigmoid,
        W=init.Orthogonal(),
    )

    return l_out
