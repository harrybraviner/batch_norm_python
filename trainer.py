#! /usr/bin/python3

import network, mnist
import time
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
                                                     fc1_size = 100, fc2_size = hidden_layer_size, fc3_size = hidden_layer_size,
                                                     output_size=10)

        self._loss_history = []

    def train_batch(self):

        im_batch, label_batch = self._training_set.getNextBatch(self._batch_size)
        #label_batch = np.array([9]*(len(label_batch)))

        #print('labels: {}'.format(label_batch))

        self._net.train_for_single_batch(im_batch, label_batch)
        self._loss_history.append(self._net._smoothed_cross_entropy_loss)

    def smoothed_training_loss(self):
        return self._loss_history[-1]

def main():
    trainer = Trainer(small_net=True)

    start_time = time.time()

    for i in range(50000):
        trainer.train_batch()
        if i%100 == 0:
            elapsed_time = time.time() - start_time
            print('wall-clock time: {}\tbatches: {}'.format(elapsed_time, i))
            print('loss: {}'.format(trainer.smoothed_training_loss()))

if __name__ == '__main__':
    main()
