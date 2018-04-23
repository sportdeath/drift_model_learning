#! /usr/local/bin/python3

import numpy as np
import tensorflow as tf

from learn import *

TENSORFLOW_GRAPH = "model.ckpt"

class TestDriftModel:

    def __init__(self):
        h = 0.008333444595336914
        training = False
        reuse = False
        self.i_ph = tf.placeholder(tf.int32, shape=(), name="i")
        self.state_batch_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, STATE_STEPS, 5), name="state_batch")
        self.control_batch_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, STATE_STEPS, 2), name="control_batch")
        self.control_check_batch_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, CHECK_STEPS, 2), name="control_check_batch")

        self.i, self.state_batch, self.control_batch = runge_kutta(self.i_ph, h, self.state_batch_ph, self.control_batch_ph, self.control_check_batch_ph, training, reuse)

        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, TENSORFLOW_GRAPH)

    def compute_f(self, i, state_batch, control_batch, control_check_batch):

        feed_dict = {}
        feed_dict[self.i_ph] = i
        feed_dict[self.control_check_batch_ph] = control_check_batch
        feed_dict[self.state_batch_ph] = state_batch
        feed_dict[self.control_batch_ph] = control_batch
        
        i, state_batch, control_batch = self.sess.run(
                (self.i, self.state_batch, self.control_batch),
                feed_dict=feed_dict)

        return i, state_batch, control_batch

if __name__ == "__main__":
    m = TestDriftModel()

    t_chunks, state_chunks, control_chunks, p_chunks = read_chunks(VALIDATION_DIR)

    for i in range(10):
        # Make a random input batch
        state_batch, control_batch, state_check_batch, control_check_batch = random_batch(
                state_chunks, control_chunks, p_chunks)

        # state by simply integrating out the state differences.
        loss_base = state_check_batch[:,-STATE_STEPS:] - state_batch

        # Use the learned model
        i = 0
        while i + 1 < CHECK_STEPS:
            i, state_batch, control_batch = m.compute_f(i, state_batch, control_batch, control_check_batch)

        loss = state_check_batch[:,-STATE_STEPS:] - state_batch
        
        print("Loss from with no model:", np.sum(np.square(loss_base)))
        print("Loss from learned model:", np.sum(np.square(loss)))
