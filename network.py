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

    def __init__(self, batch_norm, input_size = 28*28,
                       fc1_size = 100, fc2_size = 100,
                       fc3_size = 100, output_size = 10):
        if type(batch_norm) != bool:
            raise TypeError('Argument batch_norm must be a bool')

        self._batch_norm = batch_norm

        self._fc1_n = fc1_size
        self._fc2_n = fc2_size
        self._fc3_n = fc3_size
        self._input_n = input_size
        self._output_n = output_size
        self._epsilon = 1e-6 # To avoid singularities during normalization
        self._learning_rate = 1e-3

        self._non_linearity = np.vectorize(BatchNormalizableNetwork._relu)
        self._non_linearity_deriv = np.vectorize(BatchNormalizableNetwork._relu_deriv)
        #FIXME - I think I need to also define the non-linearity gradient here?

        self._setup_weights()

    def _relu(x):
        if x < 0.0:
            return 0.0
        else:
            return x

    def _relu_deriv(x):
        if x <= 0.0:
            return 0.0
        else:
            return 1.0

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
        self._all_weights = [self._fc1_W, self._fc1_b,
                             self._fc2_W, self._fc2_b,
                             self._fc3_W, self._fc3_b,
                             self._out_W, self._out_b,
                             self._fc1_beta, self._fc1_gamma,
                             self._fc2_beta, self._fc2_gamma,
                             self._fc3_beta, self._fc3_gamma]
        
        # Set the gradients to None initially
        self._fc1_W_gradient, self._fc1_b_gradient = None, None
        self._fc2_W_gradient, self._fc2_b_gradient = None, None
        self._fc3_W_gradient, self._fc3_b_gradient = None, None
        self._out_W_gradient, self._out_b_gradient = None, None
        self._fc1_beta_gradient, self._fc1_gamma_gradient = None, None
        self._fc2_beta_gradient, self._fc2_gamma_gradient = None, None
        self._fc3_beta_gradient, self._fc3_gamma_gradient = None, None
        self._all_gradients = [self._fc1_W_gradient, self._fc1_b_gradient,
                               self._fc2_W_gradient, self._fc2_b_gradient,
                               self._fc3_W_gradient, self._fc3_b_gradient,
                               self._out_W_gradient, self._out_b_gradient,
                               self._fc1_beta_gradient, self._fc1_gamma_gradient,
                               self._fc2_beta_gradient, self._fc2_gamma_gradient,
                               self._fc3_beta_gradient, self._fc3_gamma_gradient]

    def _set_weights_from_vector(self, w_in):
        i0 = 0  # Keep track of how far through w_in we are
        for w in (w for w in self._all_weights if w is not None):
            size = np.product(w.shape)
            w[:] = np.reshape(w_in[i0:i0+size], newshape=w.shape)
            i0 += size

        w_in_size = np.product(w_in.shape)
        if i0 != w_in_size:
            raise ValueError('Used {} elements to fill weights, but w_in contained {} values.'.format(i0, w_in_size))

    def _get_gradients_as_vector(self):
        weights = [self._fc1_W_gradient, self._fc1_b_gradient,
                   self._fc2_W_gradient, self._fc2_b_gradient,
                   self._fc3_W_gradient, self._fc3_b_gradient,
                   self._out_W_gradient, self._out_b_gradient,
                   self._fc1_beta_gradient, self._fc1_gamma_gradient,
                   self._fc2_beta_gradient, self._fc2_gamma_gradient,
                   self._fc3_beta_gradient, self._fc3_gamma_gradient]
        active_weights = [w for w in weights if w is not None]
        num_weights = np.sum([np.product(w.shape) for w in active_weights])
        out_v = np.zeros(shape = [num_weights])
        i0 = 0
        for w in (w for w in weights if w is not None):
            size = np.product(w.shape)
            out_v[i0:i0+size] = np.reshape(w, newshape=[-1])
            i0 += size
        return out_v

    def _single_layer_forward_prop(self, u, W, b, gamma, beta):
        if self._batch_norm:
            x_unnorm = np.matmul(u, W)
            x_mean = np.mean(x_unnorm, axis=0)
            x_delta = x_unnorm - x_mean
            x_variance = np.mean(x_delta*x_delta, axis=0)
            x_norm = (x_unnorm - x_mean) / np.sqrt(x_variance + self._epsilon)
            x_final = gamma*x_norm + beta
        else:
            x_mean = None
            x_variance = None
            x_final = np.matmul(u, W) + b

        return self._non_linearity(x_final), x_mean, x_variance

    def _forward_propagate(self, inputs):
        
        self._inputs = inputs
        self._fc1_u, self._fc1_x_mean, self._fc1_x_var = self._single_layer_forward_prop(inputs,      self._fc1_W, self._fc1_b, self._fc1_gamma, self._fc1_beta)
        self._fc2_u, self._fc2_x_mean, self._fc2_x_var = self._single_layer_forward_prop(self._fc1_u, self._fc2_W, self._fc2_b, self._fc2_gamma, self._fc2_beta)
        self._fc3_u, self._fc3_x_mean, self._fc3_x_var = self._single_layer_forward_prop(self._fc2_u, self._fc3_W, self._fc3_b, self._fc3_gamma, self._fc3_beta)
        self._out_u = np.matmul(self._fc3_u, self._out_W) + self._out_b

    def _get_losses(self, target_ix):
        y_hat = BatchNormalizableNetwork.softmax(self._out_u)
        true_y_hat = y_hat[np.arange(len(y_hat)), target_ix]
        return -np.log(true_y_hat)

    def _get_loss(self, target_ix):
        return np.mean(self._get_losses(target_ix))

    def softmax(x):
        n_out = x.shape[1]
        max_args = np.max(x, axis=1)
        safe_exp_x = np.exp(x - np.stack([max_args for _ in range(n_out)], axis=1))
        norms = np.sum(safe_exp_x, axis=1)
        return safe_exp_x / np.stack([norms for _ in range(n_out)], axis=1)

    def _make_d_dx(self, x, mean, var, d_dx_hat):
        """ Back propagates the error through the batch-normalization
        layer.
        """
        if not self._batch_norm:
            raise ValueError('Must not call _make_d_dx on un-normalized network')

        m = x.shape[0]
        d_d_sigma2 = -0.5 * np.sum(d_dx_hat * (x - mean), axis=0) * (var + self._epsilon)**(-1.5)
        d_d_mu = -1.0 * np.sum(d_dx_hat, axis=0) * (var + self._epsilon)**(-0.5) \
                -2.0 * d_d_sigma2 * np.mean(x - mean, axis=0)
        d_d_x = d_dx_hat * (var + self._epsilon)**(-0.5) \
                + 2.0 * d_d_sigma2 * (x - mean) / m \
                + d_d_mu / m
        return d_d_x

    def _back_propagate(self, target_ix):
        batch_size = target_ix.shape[0]
        delta_out = BatchNormalizableNetwork.softmax(self._out_u)
        delta_out[np.arange(len(target_ix)), target_ix] -= 1.0

        self._out_b_gradient = np.mean(delta_out, axis=0)
        # Need the below per-batch member
        dloss_dout_W = \
                np.array([np.outer(self._fc3_u[b,:], delta_out[b,:]) for b in range(batch_size)])
        self._out_W_gradient = np.mean(dloss_dout_W, axis=0)

        delta_fc3 = np.matmul(delta_out, self._out_W.T) * self._non_linearity_deriv(self._fc3_u)
        if not self._batch_norm:
            self._fc3_b_gradient = np.mean(delta_fc3, axis=0)
            self._fc3_W_gradient = np.tensordot(self._fc2_u, delta_fc3, axes=[0, 0]) / batch_size
        else:
            self._fc3_beta_gradient = np.mean(delta_fc3, axis=0)
            fc3_x_hat = (np.matmul(self._fc2_u, self._fc3_W) - self._fc3_x_mean)/np.sqrt(self._fc3_x_var + self._epsilon)
            self._fc3_gamma_gradient = np.mean(fc3_x_hat * delta_fc3, axis=0)
            # Notation of Ioffe & Szegedy, x = u.W
            delta_fc3_x = self._make_d_dx(np.matmul(self._fc2_u, self._fc3_W),
                                          self._fc3_x_mean, self._fc3_x_var, delta_fc3 * self._fc3_gamma)
            self._fc3_W_gradient = np.tensordot(self._fc2_u, delta_fc3_x, axes=[0, 0]) / batch_size

        if not self._batch_norm:
            delta_fc2 = np.matmul(delta_fc3, self._fc3_W.T) * self._non_linearity_deriv(self._fc2_u)
            self._fc2_b_gradient = np.mean(delta_fc2, axis=0)
            self._fc2_W_gradient = np.tensordot(self._fc1_u, delta_fc2, axes=[0, 0]) / batch_size
        else:
            delta_fc2 = np.matmul(delta_fc3_x, self._fc3_W.T) * self._non_linearity_deriv(self._fc2_u)
            self._fc2_beta_gradient = np.mean(delta_fc2, axis=0)
            fc2_x_hat = (np.matmul(self._fc1_u, self._fc2_W) - self._fc2_x_mean)/np.sqrt(self._fc2_x_var + self._epsilon)
            self._fc2_gamma_gradient = np.mean(fc2_x_hat * delta_fc2, axis=0)
            delta_fc2_x = self._make_d_dx(np.matmul(self._fc1_u, self._fc2_W),
                                          self._fc2_x_mean, self._fc2_x_var, delta_fc2 * self._fc2_gamma)
            self._fc2_W_gradient = np.tensordot(self._fc1_u, delta_fc2_x, axes=[0, 0]) / batch_size

        if not self._batch_norm:
            delta_fc1 = np.matmul(delta_fc2, self._fc2_W.T) * self._non_linearity_deriv(self._fc1_u)
            self._fc1_b_gradient = np.mean(delta_fc1, axis=0)
            self._fc1_W_gradient = np.tensordot(self._inputs, delta_fc1, axes=[0, 0]) / batch_size
        else:
            delta_fc1 = np.matmul(delta_fc2_x, self._fc2_W.T) * self._non_linearity_deriv(self._fc1_u)
            self._fc1_beta_gradient = np.mean(delta_fc1, axis=0)
            fc1_x_hat = (np.matmul(self._inputs, self._fc1_W) - self._fc1_x_mean)/np.sqrt(self._fc1_x_var + self._epsilon)
            self._fc1_gamma_gradient = np.mean(fc1_x_hat * delta_fc1, axis=0)
            delta_fc1_x = self._make_d_dx(np.matmul(self._inputs, self._fc1_W),
                                          self._fc1_x_mean, self._fc1_x_var, delta_fc1 * self._fc1_gamma)
            self._fc1_W_gradient = np.tensordot(self._inputs, delta_fc1_x, axes=[0, 0]) / batch_size
                                
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

    w_in_size_no_norm =   28*28*100 + 100 \
                        + 100*100   + 100 \
                        + 100*100   + 100 \
                        + 100*10    + 10
    w_in_size_with_norm =   28*28*100   \
                          + 100*100     \
                          + 100*100     \
                          + 100*10 + 10 \
                          + 100 + 100   \
                          + 100 + 100   \
                          + 100 + 100

    def test_setting_weights_no_norm(self):

        net = BatchNormalizableNetwork(False)
        w_in_size = self.w_in_size_no_norm
        w_in = np.random.uniform(size=(w_in_size))

        net._set_weights_from_vector(w_in)

        self.assertEqual(w_in[0], net._fc1_W[0, 0])
        self.assertEqual(w_in[28*28*100 + 10], net._fc1_b[10])
        out_w_offset  = w_in_size - 100*10 - 10
        self.assertEqual(w_in[out_w_offset + 5], net._out_W[0, 5])
        self.assertEqual(w_in[out_w_offset + 100*10 + 8], net._out_b[8])


    def test_setting_weights_with_norm(self):

        net = BatchNormalizableNetwork(True)
        w_in_size = self.w_in_size_with_norm
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

    def test_get_loss(self):
        net = BatchNormalizableNetwork(False)
        net._out_u = np.zeros(shape = [3, 10])
        net._out_u[0, 0] = 100.0
        net._out_u[1, 5] = -100.0
        target_y_ix = np.array([0, 5, 1])
        
        losses = net._get_losses(target_y_ix)
        expected_output = np.array([0.0, -np.log(np.exp(-100)/(np.exp(-100) + 9)), -np.log(0.1)])
        
        numpy.testing.assert_array_almost_equal(expected_output, losses)


    def check_gradients(net, segments):
        """It's possible that some gradient components will be correct
        and others won't be. In fact, it's a lot easier to get the 'later'
        (e.g. self._out_W) gradients correct that the 'earlier' components.
        
        Therefore this function breaks down the checking into named chunks.
        """

        epsilon = 1e-6
        delta_error = 1e-4
        max_failures_to_print = 5
        batch_size = 7

        weights_0 = np.random.uniform(size = sum([l for (_, (_, l)) in segments]))
        x_0 = np.random.uniform(size = [batch_size, net._input_n])
        target_ix = np.random.choice(net._output_n, size=[batch_size])
        net._set_weights_from_vector(weights_0)
        net._forward_propagate(x_0)
        net._back_propagate(target_ix)
        vector_grad = net._get_gradients_as_vector()

        failures = []

        for (name, (start_ix, length)) in segments:
            indices = range(start_ix, start_ix + length)
            segment_failures = []
            failure_count = 0
            for ix in indices:
                #print('ix: {}'.format(ix))

                weights_0[ix] += epsilon 
                net._set_weights_from_vector(weights_0)
                net._forward_propagate(x_0)
                J_plus  = net._get_loss(target_ix)
                weights_0[ix] -= 2.0*epsilon
                net._set_weights_from_vector(weights_0)
                net._forward_propagate(x_0)
                J_minus = net._get_loss(target_ix)
                weights_0[ix] += epsilon

                approx_gradient = (J_plus - J_minus) / (2.0 * epsilon)

                if np.abs(approx_gradient - vector_grad[ix]) > delta_error:
                    failure_count += 1
                    if failure_count < max_failures_to_print:
                        segment_failures.append((ix, approx_gradient, vector_grad[ix]))

            if segment_failures != []:
                failures.append((name, (start_ix, start_ix + length - 1), failure_count, segment_failures))

        if failures != []:
            error_message = 'Gradients from back propagation did not match those approximated by finite differences.\n'
            for (name, (start_ix, end_ix), failure_count, segment_failures) in failures:
                if failure_count == len(segment_failures):
                    error_message += 'In {} got {} failures:\n'.format(name, failure_count)
                else:
                    error_message += 'In {} got {} failures (printing first {}):\n'.format(name, failure_count, len(segment_failures))
                error_message += 'ix\tapprox\tback-prop\n'
                for (ix, approx_grad, vector_grad) in segment_failures:
                    error_message += '{}\t{}\t{}\n'.format(ix, approx_grad, vector_grad)
            raise AssertionError(error_message)

    def test_gradients_no_norm(self):
        net = BatchNormalizableNetwork(False, input_size=28, fc1_size=13,
                                       fc2_size=7, fc3_size=5, output_size=3)
        # Note that these are listed in backwards order!
        segment_sizes = [('out_b', 3),  ('out_W', 5*3),
                         ('fc3_b', 5),  ('fc3_W', 7*5),
                         ('fc2_b', 7),  ('fc2_W', 13*7),
                         ('fc1_b', 13), ('fc1_W', 28*13)]
        ix = sum([s for (_, s) in segment_sizes])
        segments = []
        for (name, size) in segment_sizes:
            ix -= size
            segments.append((name, (ix, size)))
        BatchNormalizableNetworkTests.check_gradients(net, segments)

    def test_gradients_with_norm(self):
        net = BatchNormalizableNetwork(True, input_size=28, fc1_size=13,
                                       fc2_size=7, fc3_size=5, output_size=3)
        # Note that these are listed in backwards order!
        segment_sizes = [('fc3_gamma', 5),  ('fc3_beta', 5),
                         ('fc2_gamma', 7),  ('fc2_beta', 7),
                         ('fc1_gamma', 13), ('fc1_beta', 13),
                         ('out_b', 3),   ('out_W', 5*3),
                         ('fc3_W', 7*5), ('fc2_W', 13*7),
                         ('fc1_W', 28*13)]
        ix = sum([s for (_, s) in segment_sizes])
        segments = []
        for (name, size) in segment_sizes:
            ix -= size
            segments.append((name, (ix, size)))
        BatchNormalizableNetworkTests.check_gradients(net, segments)
