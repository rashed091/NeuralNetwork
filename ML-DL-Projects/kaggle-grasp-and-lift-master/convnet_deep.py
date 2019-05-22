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
                output_dim,):
    l_in = layers.InputLayer(
        shape=(batch_size, num_channels, input_length),
        name='l_in',
    )

    l_sampling = SubsampleLayer(
        l_in,
        window=(None, None, 5),
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
        pad='same',
        filter_size=1,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_conv2 = Conv1DLayer(
        l_conv1,
        name='conv2',
        num_filters=8,
        pad='same',
        filter_size=1,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_pool2 = MaxPool1DLayer(
        l_conv2,
        name='pool2',
        pool_size=3,
        stride=2,
    )

    l_conv3 = Conv1DLayer(
        l_pool2,
        name='conv3',
        num_filters=32,
        pad='same',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_conv4 = Conv1DLayer(
        l_conv3,
        name='conv4',
        num_filters=16,
        pad='same',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_pool4 = MaxPool1DLayer(
        l_conv4,
        name='pool4',
        pool_size=3,
        stride=2,
    )

    l_conv5 = Conv1DLayer(
        l_pool4,
        name='conv5',
        num_filters=64,
        pad='same',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_conv6 = Conv1DLayer(
        l_conv5,
        name='conv6',
        num_filters=32,
        pad='same',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_pool6 = MaxPool1DLayer(
        l_conv6,
        name='pool6',
        pool_size=3,
        stride=2,
    )

    l_conv7 = Conv1DLayer(
        l_pool6,
        name='conv7',
        num_filters=64,
        pad='same',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_conv8 = Conv1DLayer(
        l_conv7,
        name='conv8',
        num_filters=32,
        pad='same',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_pool8 = MaxPool1DLayer(
        l_conv8,
        name='pool8',
        pool_size=3,
        stride=2,
    )

    l_dropout_dense1 = layers.DropoutLayer(
        l_pool8,
        p=0.5,
    )

    l_dense1 = layers.DenseLayer(
        l_dropout_dense1,
        num_units=64,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_dropout_dense2 = layers.DropoutLayer(
        l_dense1,
        p=0.5,
    )

    l_dense2 = layers.DenseLayer(
        l_dropout_dense2,
        num_units=64,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_out = layers.DenseLayer(
        l_dense2,
        num_units=output_dim,
        nonlinearity=nonlinearities.sigmoid,
        W=init.Orthogonal(),
    )

    return l_out
