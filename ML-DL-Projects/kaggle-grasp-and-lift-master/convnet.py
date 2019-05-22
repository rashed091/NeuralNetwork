import layers_custom
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
SubsampleLayer = layers_custom.SubsampleLayer
WindowNormLayer = layers_custom.WindowNormLayer


def build_model(batch_size,
                num_channels,
                input_length,
                output_dim,
                subsample,):
    l_in = layers.InputLayer(
        shape=(batch_size, num_channels, input_length),
        name='input',
    )

    l_sampling = SubsampleLayer(
        l_in,
        window=(None, None, 10),
        name='l_sampling',
    )

    l_window = WindowNormLayer(
        l_sampling,
        name='l_window',
    )

    l_conv1 = Conv1DLayer(
        l_window,
        name='conv1',
        num_filters=16,
        border_mode='same',
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
        num_filters=32,
        border_mode='same',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_conv3 = Conv1DLayer(
        l_conv2,
        name='conv3',
        num_filters=64,
        border_mode='same',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_pool3 = MaxPool1DLayer(
        l_conv3,
        name='pool3',
        pool_size=3,
        stride=2,
    )

    l_dropout_dense1 = layers.DropoutLayer(
        l_pool3,
        p=0.5,
    )

    l_dense1 = layers.DenseLayer(
        l_dropout_dense1,
        num_units=128,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_dropout_dense2 = layers.DropoutLayer(
        l_dense1,
        p=0.5,
    )

    l_dense2 = layers.DenseLayer(
        l_dropout_dense2,
        num_units=128,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_out = layers.DenseLayer(
        l_dense2,
        name='output',
        num_units=output_dim,
        nonlinearity=nonlinearities.sigmoid,
        W=init.Orthogonal(),
    )

    return l_out
