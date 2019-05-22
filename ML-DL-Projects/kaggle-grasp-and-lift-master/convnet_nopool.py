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
MeanSubsampleLayer = layers_custom.MeanSubsampleLayer
WindowNormLayer = layers_custom.WindowNormLayer


def build_model(batch_size,
                num_channels,
                input_length,
                output_dim):
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
        #l_in,
        name='l_window',
    )

    l_conv1_drop = layers.DropoutLayer(
        l_window,
        name='drop_conv1',
        p=0.1,
    )

    l_conv1 = Conv1DLayer(
        l_conv1_drop,
        name='conv1',
        num_filters=16,
        border_mode='valid',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_conv2 = Conv1DLayer(
        l_conv1,
        name='conv2',
        num_filters=16,
        pool_size=3,
        stride=2,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_dropout_conv3 = layers.DropoutLayer(
        l_conv2,
        name='drop_conv3',
        p=0.2,
    )

    l_conv3 = Conv1DLayer(
        l_dropout_conv3,
        name='conv3',
        num_filters=32,
        border_mode='valid',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_conv4 = Conv1DLayer(
        l_conv3,
        name='conv4',
        num_filters=32,
        pool_size=3,
        stride=2,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_dropout_conv5 = layers.DropoutLayer(
        l_conv4,
        name='drop_conv5',
        p=0.3,
    )

    l_conv5 = Conv1DLayer(
        l_dropout_conv5,
        name='conv5',
        num_filters=64,
        border_mode='valid',
        filter_size=1,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_dropout_conv6 = layers.DropoutLayer(
        l_conv5,
        name='drop_conv6',
        p=0.4,
    )

    l_conv6 = Conv1DLayer(
        l_dropout_conv6,
        name='conv6',
        num_filters=64,
        border_mode='valid',
        filter_size=1,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_conv7 = Conv1DLayer(
        l_conv6,
        name='conv7',
        num_filters=64,
        pool_size=3,
        stride=2,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_dropout_dense1 = layers.DropoutLayer(
        l_conv7,
        name='drop_dense1',
        p=0.5,
    )

    l_dense1 = layers.DenseLayer(
        l_dropout_dense1,
        name='dense1',
        num_units=128,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_dropout_dense2 = layers.DropoutLayer(
        l_dense1,
        name='drop_dense2',
        p=0.5,
    )

    l_dense2 = layers.DenseLayer(
        l_dropout_dense2,
        name='dense2',
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
