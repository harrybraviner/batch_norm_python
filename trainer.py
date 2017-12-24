#! /usr/bin/python3

import network, mnist
import time, os
import numpy as np

class Trainer:

    def __init__(self, batch_norm=False, small_net=True, output_dir = None):
        self._training_set = mnist.get_training_set()
        self._test_set = mnist.get_test_set()
        self._batch_size = 32
        self._batch_norm = batch_norm

        if small_net:
            hidden_layer_size = 10
        else:
            hidden_layer_size=100
        self._net = network.BatchNormalizableNetwork(batch_norm=batch_norm, input_size=28*28,
                                                     fc1_size = 100, fc2_size = hidden_layer_size, fc3_size = hidden_layer_size,
                                                     output_size=10)

        self._examples_trained_on = 0
        self._loss_history = []
        self._validation_loss_history = []

        if output_dir is not None:
            if os.path.exists(output_dir) == False:
                os.mkdir(output_dir)
            self._training_output_file = os.path.join(output_dir, 'training.dat')
            self._test_output_file = os.path.join(output_dir, 'test.dat')
            f = open(self._training_output_file, 'wt')
            f.write('#examples\tcross-entropy loss\n')
            f.close()
            f=open(self._test_output_file, 'wt')
            f.write('#examples\tvalidation cross-entropy\n')
            f.close()
        else:
            self._training_output_file = None
            self._test_output_file = None

    def train_batch(self):

        im_batch, label_batch = self._training_set.getNextBatch(self._batch_size)

        self._examples_trained_on += len(im_batch)
        self._net.train_for_single_batch(im_batch, label_batch)
        self._loss_history.append((self._examples_trained_on, self._net._smoothed_cross_entropy_loss))

        if self._training_output_file is not None:
            f = open(self._training_output_file, 'at')
            f.write('{}\t{}\n'.format(self._examples_trained_on, self._loss_history[-1][1]))
            f.close()

    def smoothed_training_loss(self):
        return self._loss_history[-1][1]

    def evaluate_test_set(self):
        if self._batch_norm:
            # FIXME - stats
            # Need to run the entire training set through the network to
            # produce the mean and variances for each batch-normalization layer
            raise NotImplementedError

        # Would this be faster if I broke the test set down into batches?
        im_batch, label_batch = self._test_set.getAll()
        y_hat_batch = self._net.perform_inference_for_batch(im_batch)

        validation_ce = np.mean(-np.log(y_hat_batch[np.arange(len(label_batch)), label_batch]))

        self._validation_loss_history.append((self._examples_trained_on, validation_ce))

        if self._test_output_file is not None:
            f = open(self._test_output_file, 'at')
            f.write('{}\t{}\n'.format(self._examples_trained_on, self._validation_loss_history[-1][1]))
            f.close()

    def validation_loss(self):
        return self._validation_loss_history[-1][1]

    def epochs_trained_on(self):
        return self._examples_trained_on / self._training_set.N_images

def main():
    logdir = './non_norm'
    trainer = Trainer(small_net=True, output_dir = logdir)

    start_time = time.time()

    for i in range(20000):
        trainer.train_batch()
        if i%100 == 0:
            print('loss: {}'.format(trainer.smoothed_training_loss()))
        if i%1000 == 0:
            elapsed_time = time.time() - start_time
            print('Wall clock time: {}'.format(elapsed_time))
            print('Epochs trained on: {}'.format(trainer.epochs_trained_on()))
            trainer.evaluate_test_set()
            print('validation loss: {}'.format(trainer.validation_loss()))

if __name__ == '__main__':
    main()
