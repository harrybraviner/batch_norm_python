#! /usr/bin/python3

import network, mnist
import numpy as np

class Trainer:

    def __init__(self, batch_norm=False, small_net=True):
        self._training_set = mnist.get_training_set()
        self._test_set = mnist.get_test_set()
        self._batch_size = 32

        if small_net:
            hidden_layer_size = 10
        else:
            hidden_layer_size=100
        self._net = network.BatchNormalizableNetwork(batch_norm=batch_norm, input_size=28*28,
                                                     fc1_size = 100, fc2_size = 100, fc3_size = 100,
                                                     output_size=10)

    def train_batch(self):

        im_batch, label_batch = self._training_set.getNextBatch(self._batch_size)
        im_batch = np.reshape(im_batch, newshape = [-1, 28*28])

        self._net.train_for_single_batch(im_batch, label_batch)

def main():
    trainer = Trainer()

    trainer.train_batch()

if __name__ == '__main__':
    main()
