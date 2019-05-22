import lasagne
import theano
import theano.tensor as T

from lasagne import layers


def create_iter_funcs_train(lr, mntm, l_out):
    X = T.tensor3('x')
    y = T.imatrix('y')
    X_batch = T.tensor3('x_batch')
    y_batch = T.imatrix('y_batch')

    train_output = layers.get_output(l_out, X_batch)
    train_loss = T.mean(T.nnet.binary_crossentropy(train_output, y_batch))

    all_params = layers.get_all_params(l_out)
    updates = lasagne.updates.nesterov_momentum(
        train_loss, all_params, lr, mntm)

    train_iter = theano.function(
        inputs=[theano.Param(X_batch),
                theano.Param(y_batch)],
        outputs=[train_loss, train_output],
        updates=updates,
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    return train_iter


def create_iter_funcs_valid(l_out):
    X = T.tensor3('x')
    y = T.imatrix('y')
    X_batch = T.tensor3('x_batch')
    y_batch = T.imatrix('y_batch')

    valid_output = layers.get_output(l_out, X_batch, deterministic=True)
    valid_loss = T.mean(T.nnet.binary_crossentropy(valid_output, y_batch))

    valid_iter = theano.function(
        inputs=[theano.Param(X_batch),
                theano.Param(y_batch)],
        outputs=[valid_loss, valid_output],
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    return valid_iter


def create_iter_funcs_test(l_out):
    X = T.tensor3('x')
    X_batch = T.tensor3('x_batch')

    test_output = layers.get_output(l_out, X_batch, deterministic=True)

    test_iter = theano.function(
        inputs=[theano.Param(X_batch)],
        outputs=test_output,
        givens={
            X: X_batch,
        },
    )

    return test_iter
