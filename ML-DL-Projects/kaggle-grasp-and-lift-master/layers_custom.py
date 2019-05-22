#!/usr/bin/env python

import numpy as np
import theano.tensor as T
from lasagne import layers


class WindowNormLayer(layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(WindowNormLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        X_min = T.min(input, axis=2).reshape((-1, input.shape[1], 1))
        X_max = T.max(input, axis=2).reshape((-1, input.shape[1], 1))

        return (input - X_min) / (X_max - X_min)


# given a one-dimensional signal, this layer subsamples the signal
# between the specified time-steps, ensuring that the final time-step
# that is sampled is always the final one in the input signal
class SubsampleLayer(layers.Layer):
    def __init__(self, incoming, window, **kwargs):
        super(SubsampleLayer, self).__init__(incoming, **kwargs)
        if not isinstance(window, slice):
            self.window = slice(*window)
        else:
            self.window = window

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        # compute the output window length based on the range of the window
        if self.window.stop is None and self.window.start is None:
            output_shape[2] = input_shape[2] / self.window.step
        elif self.window.stop is None:
            output_shape[2] = (input_shape[2] -
                               self.window.start) / self.window.step
        elif self.window.start is None:
            output_shape[2] = self.window.stop / self.window.step
        else:
            output_shape[2] = (self.window.stop -
                               self.window.start) / self.window.step
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        start = self.window.start
        stop = self.window.stop
        step = self.window.step
        return input[:, :, start:stop][:, :, ::-1][:, :, ::step][:, :, ::-1]


class MeanSubsampleLayer(layers.Layer):
    def __init__(self, incoming, window, **kwargs):
        super(MeanSubsampleLayer, self).__init__(incoming, **kwargs)
        if not isinstance(window, slice):
            self.window = slice(*window)
        else:
            self.window = window

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        # compute the output window length based on the range of the window
        if self.window.stop is None and self.window.start is None:
            output_shape[2] = input_shape[2] / self.window.step
        elif self.window.stop is None:
            output_shape[2] = (input_shape[2] -
                               self.window.start) / self.window.step
        elif self.window.start is None:
            output_shape[2] = self.window.stop / self.window.step
        else:
            output_shape[2] = (self.window.stop -
                               self.window.start) / self.window.step
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return T.mean(input.reshape((input.shape[0],
                                     input.shape[1],
                                     input.shape[2] / self.window.step,
                                     self.window.step)), axis=3)


def run_subsample_tests():
    l_in = layers.InputLayer(shape=(64, 32, 2000))
    l_sample = SubsampleLayer(l_in, window=(None, 1000, 10))

    X = np.random.normal(0, 1, (64, 32, 2000))
    expected_output_shape = (64, 32, 100)
    actual_output_shape = l_sample.get_output_shape_for(X.shape)
    assert expected_output_shape == actual_output_shape, '%r != %r' % (
        expected_output_shape, actual_output_shape)

    # test None:1000
    expected_output = X[:, :, None:1000][:, :, ::-1][:, :, ::10][:, :, ::-1]
    actual_output = l_sample.get_output_for(X)
    assert expected_output.shape == actual_output.shape, '%r != %r' % (
        expected_output_shape, actual_output.shape)
    assert (expected_output == actual_output).all(), 'bad subsampling'

    # test 1000:None
    l_sample = SubsampleLayer(l_in, window=(1000, None, 10))
    expected_output = X[:, :, 1000:None][:, :, ::-1][:, :, ::10][:, :, ::-1]
    actual_output = l_sample.get_output_for(X)
    assert expected_output.shape == actual_output.shape, '%r != %r' % (
        expected_output_shape, actual_output.shape)
    assert (expected_output == actual_output).all(), 'bad subsampling'

    # test None:None
    l_sample = SubsampleLayer(l_in, window=(None, None, 10))
    expected_output = X[:, :, None:None][:, :, ::-1][:, :, ::10][:, :, ::-1]
    actual_output = l_sample.get_output_for(X)
    assert expected_output.shape == actual_output.shape, '%r != %r' % (
        expected_output_shape, actual_output.shape)
    assert (expected_output == actual_output).all(), 'bad subsampling'

    X = np.arange(2 * 4 * 10).reshape(2, 4, 10)
    l_in = layers.InputLayer(shape=(2, 4, 10))
    l_sample = SubsampleLayer(l_in, window=(None, 5, 3))
    expected_output = X[:, :, (1, 4)]
    actual_output = l_sample.get_output_for(X)
    assert (expected_output == actual_output).all(), 'bad subsampling'

    X = np.arange(2 * 4 * 10).reshape(2, 4, 10)
    l_in = layers.InputLayer(shape=(2, 4, 10))
    l_sample = SubsampleLayer(l_in, window=(1, None, 4))
    expected_output = X[:, :, (1, 5, 9)]
    actual_output = l_sample.get_output_for(X)
    assert (expected_output == actual_output).all(), 'bad subsampling'

    X = np.arange(2 * 4 * 10).reshape(2, 4, 10)
    l_in = layers.InputLayer(shape=(2, 4, 10))
    l_sample = SubsampleLayer(l_in, window=(1, None, 4))
    expected_output = X[:, :, (1, 5, 9)]
    actual_output = l_sample.get_output_for(X)
    assert (expected_output == actual_output).all(), 'bad subsampling'


def run_window_tests():
    import batching
    X = np.random.normal(0, 1, (256, 32, 200))
    l_in = layers.InputLayer(shape=(256, 32, 200))
    l_window = WindowNormLayer(l_in)
    expected_output = np.empty(X.shape, dtype=np.float32)
    for i in range(0, 256):
        expected_output[i, ...] = batching.normalize_window(X[i, ...])
    actual_output = l_window.get_output_for(X)
    #assert (expected_output == actual_output).all()
    assert np.allclose(expected_output, actual_output.eval(),
                       atol=1e-05, rtol=1e-05)


def run_mean_tests():
    X = np.random.normal(0, 1, (256, 32, 2000))
    l_in = layers.InputLayer(shape=(256, 32, 2000))
    l_mean = MeanSubsampleLayer(l_in, window=(None, None, 10))
    expected_output = np.mean(X.reshape(256, 32, 200, 10), axis=3)
    actual_output = l_mean.get_output_for(X).eval()
    assert np.allclose(expected_output, actual_output,
                       atol=1e-05, rtol=1e-05)


if __name__ == '__main__':
    #print('running SubsampleLayer tests')
    #run_subsample_tests()
    #print('running WindowNormLayer tests')
    #run_window_tests()
    print('running MeanSubsampleLayer tests')
    run_mean_tests()
