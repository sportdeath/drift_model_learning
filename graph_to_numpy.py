import numpy as np
import tensorflow as tf

import params
import learn
import time_stepping

TENSORFLOW_GRAPH = "model/model.ckpt"

class DriftModel:

    def __init__(self):
        self.states = tf.placeholder(tf.float32, shape=(params.STATE_STEPS*params.STATES), name="states")
        self.controls = tf.placeholder(tf.float32, shape=(params.CONTROLS), name="controls")
        state_batch = tf.reshape(self.states, (1, params.STATE_STEPS, params.STATES))
        control_batch = tf.reshape(self.controls, (1, 1, params.CONTROLS))

        # Evaluate f at the point
        h = 0.008333444595336914
        with tf.variable_scope("time_stepping"):
            dstate_batch = learn.f(h, state_batch, control_batch, training=False, reuse=False)
        self.dstates = tf.reshape(dstate_batch, (params.STATE_STEPS*params.STATES,), name="dstates")

        # Differentiate f with respect to the control inputs
        As = []
        Bs = []
        for i in range(params.STATE_STEPS*params.STATES):
            As.append(tf.gradients(self.dstates[i], self.states))
            Bs.append(tf.gradients(self.dstates[i], self.controls))
        self.A = tf.concat(As, axis=0, name="A")
        self.B = tf.concat(Bs, axis=0, name="B")

        # Restore the graph
        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, TENSORFLOW_GRAPH)

    def f(self, states, controls):
        feed_dict = {}
        feed_dict[self.states] = states
        feed_dict[self.controls] = controls

        return self.sess.run(self.dstates, feed_dict=feed_dict)

    def AB(self, states, controls):
        feed_dict = {}
        feed_dict[self.states] = states
        feed_dict[self.controls] = controls

        return self.sess.run((self.A,self.B), feed_dict=feed_dict)

if __name__ == "__main__":
    from timeit import default_timer as timer

    states = np.zeros(params.STATE_STEPS * params.STATES)
    controls = np.zeros(params.CONTROLS)
    print("Loading model...")
    dm = DriftModel()
    print("Loaded.")

    NUM_TRIALS = 1000
    start = timer()
    for i in range(NUM_TRIALS):
        dstates = dm.f(states, controls)
    end = timer()
    print("Computed f in", (end - start)/NUM_TRIALS, "seconds.")
    print(dstates)

    start = timer()
    for i in range(NUM_TRIALS):
        A, B = dm.AB(states, controls)
    end = timer()
    print("Computed A, B in", (end - start)/NUM_TRIALS, "seconds.")
    print(A)
    print(B)
