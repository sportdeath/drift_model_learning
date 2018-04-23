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
        self.state_batch_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, STATE_STEPS, 5), name="state_batch")
        self.control_batch_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, STATE_STEPS, 2), name="control_batch")

        with tf.variable_scope("forward_euler_loss"):
            # Normalize
            origin_batch = tf.zeros((BATCH_SIZE, 1, 3))
            origin_batch = normalize_batch(origin_batch, self.state_batch_ph[:, -1])
            state_batch = normalize_batch(self.state_batch_ph, self.state_batch_ph[:, -1])

            prediction = state_batch[:,-1] + h * f(state_batch, self.control_batch_ph, training, reuse)
            prediction = tf.expand_dims(prediction,axis=1)

            # Unnormalize the prediction
            prediction_unnormalized = normalize_batch(prediction, origin_batch)
            self.prediction = prediction_unnormalized[:,0]

        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, TENSORFLOW_GRAPH)

    def compute_f(self, state_batch, control_batch):

        feed_dict = {}
        feed_dict[self.state_batch_ph] = state_batch
        feed_dict[self.control_batch_ph] = control_batch
        
        prediction = self.sess.run(self.prediction, feed_dict=feed_dict)

        return prediction

if __name__ == "__main__":
    m = TestDriftModel()

    t_chunks, state_chunks, control_chunks, p_chunks = read_chunks(VALIDATION_DIR)

    for i in range(10):
        # Make a random input batch
        state_batch, control_batch, state_check_batch, control_check_batch = random_batch(
                state_chunks, control_chunks, p_chunks)

        # state by simply integrating out the state differences.
        prediction_base = state_batch[:,-1] + len(state_check_batch) * (state_batch[:,-1] - state_batch[:,-2])

        # Use the learned model
        for i in range(len(state_check_batch)):
            prediction = m.compute_f(state_batch, control_batch)
            print(i,prediction[0])
            state_batch = np.concatenate((state_batch[:,1:], np.expand_dims(prediction,axis=1)),axis=1)
            control_batch = np.concatenate((control_batch[:,1:], np.expand_dims(control_check_batch[:,i], axis=1)),axis=1)

        loss_base = state_check_batch[:,-1] - prediction_base
        loss = state_check_batch[:,-1] - prediction
        
        print("Loss from velocity integration:", np.sum(np.square(loss_base)))
        print("Loss from learned model:", np.sum(np.square(loss)))
