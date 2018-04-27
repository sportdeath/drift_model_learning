#! /usr/local/bin/python3

import numpy as np
import tensorflow as tf

import learn
import params
import read_data
import process_data
import time_stepping

TENSORFLOW_GRAPH = "model.ckpt"

class TestDriftModel:

    def __init__(self):
        h = 0.008333444595336914
        training = False
        reuse = False
        self.i_ph = tf.placeholder(tf.int32, shape=(), name="i")
        self.state_batch_ph = tf.placeholder(tf.float32, shape=(params.BATCH_SIZE, params.STATE_STEPS, params.STATES), name="state_batch")
        self.control_batch_ph = tf.placeholder(tf.float32, shape=(params.BATCH_SIZE, params.STATE_STEPS, params.CONTROLS), name="control_batch")
        self.control_check_batch_ph = tf.placeholder(tf.float32, shape=(params.BATCH_SIZE, params.CHECK_STEPS, params.CONTROLS), name="control_check_batch")

        ts = time_stepping.RungeKutta(learn.f)
        self.i, self.state_batch, self.control_batch = ts.integrate(self.i_ph, h, self.state_batch_ph, self.control_batch_ph, self.control_check_batch_ph, training, reuse)

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

    t_chunks, state_chunks, control_chunks, p_chunks = read_data.read_chunks(params.VALIDATION_DIR)

    for i in range(10):
        # Make a random input batch
        state_batch, control_batch, state_check_batch, control_check_batch = process_data.random_batch(
                state_chunks, control_chunks, p_chunks)

        # state by simply integrating out the state differences.
        loss_base = state_check_batch[:,-1] - (state_batch[:,-1] + params.CHECK_STEPS * (state_batch[:,-1] - state_batch[:,-2]))

        # Use the learned model
        i, state_batch, control_batch = m.compute_f(0, state_batch, control_batch, control_check_batch)

        # loss = state_check_batch[:,-STATE_STEPS:] - state_batch
        loss = state_check_batch[:,-1] - state_batch[:,-1]
        
        print("Loss from with no model:", np.sum(np.square(loss_base)))
        print("Loss from learned model:", np.sum(np.square(loss)))

    state_batch = np.zeros((params.BATCH_SIZE, params.STATE_STEPS, params.STATES),dtype=np.float32)
    control_batch = np.zeros((params.BATCH_SIZE, params.STATE_STEPS, params.CONTROLS))
    control_batch[:,:,1] = 0.52
    control_check_batch = np.zeros((params.BATCH_SIZE, params.CHECK_STEPS, params.CONTROLS))
    control_check_batch[:,:,1] = 0.52

    for i in range(1000):
        print(state_batch[0])
        i, next_state_batch, control_batch = m.compute_f(0, state_batch, control_batch, control_check_batch)
        state_batch = np.concatenate((state_batch[:,i:], next_state_batch[:,-i:]), axis=1)
