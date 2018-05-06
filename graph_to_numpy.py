import tensorflow as tf
import tfdeploy as tfd

import params
import learn
import time_stepping

TENSORFLOW_GRAPH = "model/model.ckpt"
NUMPY_MODEL = "model/model.pkl"

def graph_to_numpy():
    # Initialize inputs
    states = tf.placeholder(tf.float32, shape=(params.STATE_STEPS, params.STATES), name="states")
    controls = tf.placeholder(tf.float32, shape=(params.STATE_STEPS, params.CONTROLS), name="controls")
    state_batch = tf.reshape(states, (1, params.STATE_STEPS, params.STATES))
    control_batch = tf.reshape(controls, (1, params.STATE_STEPS, params.CONTROLS))

    # Evaluate f
    h = 0.008333444595336914
    with tf.variable_scope("runge_kutta"):
        dstate_batch = learn.f(h, state_batch, control_batch, training=False, reuse=False)
    dstates = tf.reshape(dstate_batch, (params.STATE_STEPS*params.STATES,), name="dstates")

    # Differentiate f with respect to the control inputs
    As = []
    Bs = []
    for i in range(params.STATE_STEPS*params.STATES):
        As.append(tf.gradients(dstates[i], states))
        Bs.append(tf.gradients(dstates[i], controls))
        print(dstates[i])
        print(As)

    A = tf.concat(As, axis=0, name="A")
    B = tf.concat(Bs, axis=0, name="B")

    sess = tf.Session()
    tf.train.Saver().restore(sess, TENSORFLOW_GRAPH)

    model = tfd.Model()
    model.add(dstates, sess)
    model.add(A, sess)
    model.add(B, sess)
    model.save(NUMPY_MODEL)

if __name__ == "__main__":
    graph_to_numpy()
