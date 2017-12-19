#! /usr/bin/python3
import numpy as np
import numpy.testing
import unittest

class BatchNormalizableNetwork:
    """Relatively simple network.
    Three fully connected hidden layers of 100
    neurons each (same as Ioffe & Szegedy, 2015).

    Batch normalization is applied to a layer's inputs.
    """

    def __init__(self, batch_norm):
        if type(batch_norm) != bool:
            raise TypeError('Argument batch_norm must be a bool')

        self._batch_norm = batch_norm

        self._fc1_n = 100
        self._fc2_n = 100
        self._fc3_n = 100
        self._input_n = 28*28
        self._output_n = 10
        self._epsilon = 1e-6 # To avoid singularities during normalization
        self._learning_rate = 1e-3

        self._non_linearity = BatchNormalizableNetwork._relu
        #FIXME - I think I need to also define the non-linearity gradient here?

        self._setup_weights()

    def _relu(x):
        if x < 0.0:
            return 0.0
        else:
            return x

    def _make_weights(shape):
        return np.random.uniform(low=-0.01, high=0.01, size=shape)

    def _make_biases(shape):
        return np.random.uniform(low=0.0, high=0.01, size=shape)

    def _setup_weights(self):
        self._fc1_W = BatchNormalizableNetwork._make_weights([self._input_n, self._fc1_n])
        self._fc2_W = BatchNormalizableNetwork._make_weights([self._fc1_n,   self._fc2_n])
        self._fc3_W = BatchNormalizableNetwork._make_weights([self._fc2_n,   self._fc3_n])
        self._out_W = BatchNormalizableNetwork._make_weights([self._fc3_n,   self._output_n])
        self._out_b = BatchNormalizableNetwork._make_biases([self._output_n])
        
        if self._batch_norm:
            self._fc1_beta  = BatchNormalizableNetwork._make_biases([self._fc1_n])
            self._fc1_gamma = BatchNormalizableNetwork._make_weights([self._fc1_n])
            self._fc2_beta  = BatchNormalizableNetwork._make_biases([self._fc2_n])
            self._fc2_gamma = BatchNormalizableNetwork._make_weights([self._fc2_n])
            self._fc3_beta  = BatchNormalizableNetwork._make_biases([self._fc3_n])
            self._fc3_gamma = BatchNormalizableNetwork._make_weights([self._fc3_n])
            self._fc1_b = None
            self._fc2_b = None
            self._fc3_b = None
        else:
            self._fc1_beta  = None
            self._fc1_gamma = None
            self._fc2_beta  = None
            self._fc2_gamma = None
            self._fc3_beta  = None
            self._fc3_gamma = None
            self._fc1_b = BatchNormalizableNetwork._make_biases([self._fc1_n])
            self._fc2_b = BatchNormalizableNetwork._make_biases([self._fc2_n])
            self._fc3_b = BatchNormalizableNetwork._make_biases([self._fc3_n])

    def _set_weights_from_vector(self, w_in):
        weights = [self._fc1_W, self._fc1_b,
                   self._fc2_W, self._fc2_b,
                   self._fc3_W, self._fc3_b,
                   self._out_W, self._out_b,
                   self._fc1_beta, self._fc1_gamma,
                   self._fc2_beta, self._fc2_gamma,
                   self._fc3_beta, self._fc3_gamma]
        i0 = 0  # Keep track of how far through w_in we are
        for w in (w for w in weights if w is not None):
            size = np.product(w.shape)
            w[:] = np.reshape(w_in[i0:i0+size], newshape=w.shape)
            i0 += size

        w_in_size = np.product(w_in.shape)
        if i0 != w_in_size:
            raise ValueError('Used {} elements to fill weights, but w_in contained {} values.'.format(i0, w_in_size))

    def _get_gradients_as_vector(self):
        raise NotImplementedError

    def _single_layer_forward_prop(self, u, W, b, gamma, beta):
        if self._batch_norm:
            x_unnorm = np.matmul(u, W)
            x_mean = np.mean(x_unnorm, axis=0)
            x_delta = x_unnorm - x_mean
            x_variance = np.mean(x_delta*x_delta, axis=0)
            x_norm = (x_unnorm - x_mean) / np.sqrt(x_variance + self._epsilon)
            x_final = gamma*x_norm + beta_x_norm
        else:
            x_final = np.matmul(u, W) + b

        return self._non_linearity(x_final)

    def _forward_propagate(self, inputs):
        
        self._fc1_u = self._single_layer_forward_prop(inputs,      self._fc1_W, self._fc1_b, self._fc1_gamma, self._fc1_beta)
        self._fc2_u = self._single_layer_forward_prop(self._fc1_u, self._fc2_W, self._fc2_b, self._fc2_gamma, self._fc2_beta)
        self._fc3_u = self._single_layer_forward_prop(self._fc2_u, self._fc3_W, self._fc3_b, self._fc3_gamma, self._fc3_beta)
        self._out_u = np.matmul(self._fc3_u, self._out_W) + self._out_b

    def _get_loss(self, y_target):
        
        raise NotImplementedError

    def softmax(x):
        n_out = x.shape[1]
        max_args = np.max(x, axis=1)
        safe_exp_x = np.exp(x - np.stack([max_args for _ in range(n_out)], axis=1))
        norms = np.sum(safe_exp_x, axis=1)
        return safe_exp_x / np.stack([norms for _ in range(n_out)], axis=1)

    def _back_propagate(self, y_target):
        raise NotImplementedError

    def _take_gradient_step(self):
        self._fc1_W += -self._learning_rate*self._fc1_W_gradient
        self._fc2_W += -self._learning_rate*self._fc2_W_gradient
        self._fc3_W += -self._learning_rate*self._fc3_W_gradient
        self._out_W += -self._learning_rate*self._out_W_gradient
        self._out_b += -self._learning_rate*self._out_b_gradient
        if self._batch_norm:
            self._fc1_beta += -self._learning_rate*self._fc1_beta_gradient
            self._fc1_gamma += -self._learning_rate*self._fc1_gamma_gradient
            self._fc2_beta += -self._learning_rate*self._fc2_beta_gradient
            self._fc2_gamma += -self._learning_rate*self._fc2_gamma_gradient
            self._fc3_beta += -self._learning_rate*self._fc3_beta_gradient
            self._fc3_gamma += -self._learning_rate*self._fc3_gamma_gradient
        else:
            self._fc1_b += -self._learning_rate*self._fc1_b_gradient
            self._fc2_b += -self._learning_rate*self._fc2_b_gradient
            self._fc3_b += -self._learning_rate*self._fc3_b_gradient

    def train_for_single_batch(self, inputs, y_target):
        self._forward_propagate(inputs)
        self._back_propagate(y_target)
        self._take_gradient_step()
            
    # FIXME - remember that forward propagation during training and inference are different
    #         since inference-time uses the statistics of the whole training set
    #         Do I need to know E[x] at input to layer 1 for the whole training set before I can compute the
    #         inputs to layer 2 for the whole training set? I think I do.
    #         So I need to 'forward propagate' 3 times for the whole training set! But only once, just before
    #         inference.

class BatchNormalizableNetworkTests(unittest.TestCase):

    def test_setting_weights_no_norm(self):

        net = BatchNormalizableNetwork(False)
        w_in_size =   28*28*100 + 100 \
                    + 100*100   + 100 \
                    + 100*100   + 100 \
                    + 100*10    + 10
        w_in = np.random.uniform(size=(w_in_size))

        net._set_weights_from_vector(w_in)

        self.assertEqual(w_in[0], net._fc1_W[0, 0])
        self.assertEqual(w_in[28*28*100 + 10], net._fc1_b[10])
        out_w_offset  = w_in_size - 100*10 - 10
        self.assertEqual(w_in[out_w_offset + 5], net._out_W[0, 5])
        self.assertEqual(w_in[out_w_offset + 100*10 + 8], net._out_b[8])


    def test_setting_weights_with_norm(self):

        net = BatchNormalizableNetwork(True)
        w_in_size =   28*28*100   \
                    + 100*100     \
                    + 100*100     \
                    + 100*10 + 10 \
                    + 100 + 100   \
                    + 100 + 100   \
                    + 100 + 100
        w_in = np.random.uniform(size=(w_in_size))

        net._set_weights_from_vector(w_in)

        self.assertEqual(w_in[0], net._fc1_W[0, 0])
        self.assertEqual(w_in[28*28*100 + 105], net._fc2_W[1, 5])
        out_w_offset  = 28*28*100 + 100*100*2
        self.assertEqual(w_in[out_w_offset + 5], net._out_W[0, 5])
        fc2_beta_offset = w_in_size - 4*100
        self.assertEqual(w_in[fc2_beta_offset + 50], net._fc2_beta[50])
        self.assertEqual(w_in[fc2_beta_offset + 150], net._fc2_gamma[50])
        
    def test_softmax(self):

        out_u = np.array([[1.0, 10.0],
                          [-1.0, 0.0]])

        expected_output = np.array([[np.exp(1.0)/(np.exp(1.0) + np.exp(10.0)), np.exp(10.0)/(np.exp(1.0) + np.exp(10.0))],
                                    [np.exp(-1.0)/(np.exp(-1.0) + np.exp(0.0)), np.exp(0.0)/(np.exp(-1.0) + np.exp(0.0))]])

        numpy.testing.assert_array_almost_equal(expected_output, BatchNormalizableNetwork.softmax(out_u), decimal=8)

#    def test_check_gradients_no_norm(self):
#
#        w_in_size =   28*28*100 + 100 \
#                    + 100*100   + 100 \
#                    + 100*100   + 100 \
#                    + 100*10    + 10
#        inputs = np.random.uniform(size=[28*28])
#        weights = np.random.uniform(size=w_in_size)
#        target_y = np.zeros(shape=10)
#        target_y[3] = 1.0
#
#        net._set_weights_from_vector(weights)
#        net.
