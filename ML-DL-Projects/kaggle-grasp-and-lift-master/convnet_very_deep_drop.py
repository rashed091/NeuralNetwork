import subsample
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
SubsampleLayer = subsample.SubsampleLayer


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
        window=(None, None, subsample),
        name='l_sampling',
    )

    l_conv1 = Conv1DLayer(
        l_sampling,
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

    l_dropout_conv2 = layers.DropoutLayer(
        l_pool1,
        name='drop_conv2',
        p=0.2,
    )

    l_conv2 = Conv1DLayer(
        l_dropout_conv2,
        name='conv2',
        num_filters=16,
        border_mode='valid',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_dropout_conv3 = layers.DropoutLayer(
        l_conv2,
        name='drop_conv2',
        p=0.2,
    )

    l_conv3 = Conv1DLayer(
        l_dropout_conv3,
        name='conv2',
        num_filters=16,
        border_mode='valid',
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

    l_dropout_conv4 = layers.DropoutLayer(
        l_pool3,
        name='drop_conv4',
        p=0.3,
    )

    l_conv4 = Conv1DLayer(
        l_dropout_conv4,
        name='conv4',
        num_filters=32,
        border_mode='valid',
        filter_size=3,
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
        name='conv4',
        num_filters=32,
        border_mode='valid',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_pool5 = MaxPool1DLayer(
        l_conv5,
        name='pool5',
        pool_size=3,
        stride=2,
    )

    l_dropout_conv6 = layers.DropoutLayer(
        l_pool5,
        name='drop_conv4',
        p=0.4,
    )

    l_conv6 = Conv1DLayer(
        l_dropout_conv6,
        name='conv6',
        num_filters=64,
        border_mode='valid',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_dropout_conv7 = layers.DropoutLayer(
        l_conv6,
        name='drop_conv7',
        p=0.4,
    )

    l_conv7 = Conv1DLayer(
        l_dropout_conv7,
        name='conv7',
        num_filters=64,
        border_mode='valid',
        filter_size=3,
        nonlinearity=nonlinearities.rectify,
        W=init.Orthogonal(),
    )

    l_pool7 = MaxPool1DLayer(
        l_conv7,
        name='pool7',
        pool_size=3,
        stride=2,
    )

    l_dropout_dense1 = layers.DropoutLayer(
        l_pool7,
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
