import numpy as np
import tensorflow as tf

import params
import learn
import time_stepping

TENSORFLOW_GRAPH = "model/model.ckpt"

if __name__ == "__main__":
    states = tf.placeholder(tf.float32, shape=(params.STATE_STEPS*params.STATES), name="states")
    controls = tf.placeholder(tf.float32, shape=(params.CONTROLS), name="controls")
    state_batch = tf.reshape(states, (1, params.STATE_STEPS, params.STATES))
    control_batch = tf.reshape(controls, (1, 1, params.CONTROLS))

    # Evaluate f at the point
    h = 0.008333444595336914
    with tf.variable_scope("time_stepping"):
        dstate_batch = learn.f(h, state_batch, control_batch, training=False, reuse=False)
    dstates = tf.reshape(dstate_batch, (params.STATE_STEPS*params.STATES,), name="dstates")

    # Differentiate f with respect to the control inputs
    # As = []
    # Bs = []
    # for i in range(params.STATE_STEPS*params.STATES):
        # As.append(tf.gradients(self.dstates[i], self.states))
        # Bs.append(tf.gradients(self.dstates[i], self.controls))
    # self.A = tf.concat(As, axis=0, name="A")
    # self.B = tf.concat(Bs, axis=0, name="B")

    # Restore the graph
    sess = tf.Session()
    tf.train.write_graph(sess.graph, "model", "f.pbtxt")

    with open("model/f.config.pbtxt", 'w') as config:
        config.write(
"""feed {
    id {node_name: "states"}
    shape {
        dim {size: """ + str(params.STATE_STEPS * params.STATES) + """}
    }
}

feed {
    id {node_name: "controls"}
    shape {
        dim {size: """ + str(params.CONTROLS) + """}
    }
}

fetch {
    id {node_name: "dstates"}
}
""")
