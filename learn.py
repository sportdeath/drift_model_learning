#!/usr/bin/env python3

"""
Train a neural network to learn
drift
"""

import os
import csv
import numpy as np
import tensorflow as tf

LOG_DIR = "tmp/drifter/rk2/"

STATES = 5
CONTROLS = 2

"""
The number of states to check
"""
STATE_STEPS = 5

"""
The number of future states to verify.
"""
CHECK_STEPS = 2

"""
The number of units and the 
activation function used at the
output of each layer of the network
"""
LAYER_UNITS = [STATES * STATE_STEPS]
ACTIVATIONS = [None]

"""
The integer factor to downsample
the data. Default rate is 120hz
"""
DOWNSAMPLE = 4

"""
Percentage of the time that
a node is not dropped out.
"""
DROPOUT = 0.7

"""
The number of elements in a training batch.
"""
BATCH_SIZE = 20
POSITION_SCALING = 0.1
THETA_SCALING = 0.1
RPM_SCALING = 20000.
VOLTAGE_SCALING = 10.
STD_DEV = 0.001
TRAIN_DIR = "./train/"
VALIDATION_DIR = "./validation/"
LEARNING_RATE_START = 0.0004
LEARNING_RATE_END = 0.0004
LEARNING_RATE_END_STEPS = 1000000
LEARNING_RATE_POWER = 0.5

def read_chunks(directory):
    t_chunks = []
    state_chunks = []
    control_chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".bag.csv"):
            f = os.path.join(directory, filename)
            t, state, control = read_bag_csv_file(f)

            t_chunks.append(t)
            state_chunks.append(state)
            control_chunks.append(control)

    p_chunks = np.array([len(chunk) for chunk in state_chunks])
    p_chunks = p_chunks/float(np.sum(p_chunks))

    return t_chunks, state_chunks, control_chunks, p_chunks

def read_bag_csv_file(file_path):
    with open(file_path, 'r') as bag_file:
        reader = csv.reader(bag_file)

        data = []
        for row in reader:
            if row[0][0] == '#':
                continue
            data.append(row)

        data = np.array(data, dtype=np.float64)

        # Extract t, state, control
        t = data[:, 0]
        t -= t[0]
        controls = data[:, 1:3]
        states = data[:, 3:8]

        # Reorder the observed states to be
        # (x, y, theta, omega, V)
        states = states[:,[2, 3, 4, 0, 1]]

        # Change theta = 0 to point in
        # the positive x direction
        states[:, 2] += np.pi/2.

        # Unwrap the angles
        states[:, 2] = np.unwrap(states[:, 2])

        # Scale
        controls[:,0] /= RPM_SCALING
        states[:, :2] /= POSITION_SCALING
        states[:, 2] /= THETA_SCALING
        states[:, 3] /= RPM_SCALING
        states[:, 4] /= VOLTAGE_SCALING

    return t, states, controls

def plot_time(t):
    import matplotlib.pyplot as plt
    s = np.arange(len(t))
    plt.figure()
    plt.title('Time')
    plt.plot(s, t)
    plt.show()

def plot_state_poses(states, bounding_box=1.):
    """
    Plots the poses of a state

    Args:
        states: A numpy array where the last dimension
            is (x, y, theta, omega, V)
    """
    
    import matplotlib.pyplot as plt

    while len(states.shape) > 2:
        states = states[0]

    # If there are a whole bunch of states,
    # down sample them
    if len(states) > 200:
        states = states[::10]

    x = states[:, 0]
    y = states[:, 1]
    u = np.cos(states[:, 2])
    v = np.sin(states[:, 2])

    plt.figure()
    plt.title('State poses')
    plt.quiver(x, y, u, v, scale=20., headwidth=3., width=0.002)
    plt.ylim((-bounding_box, bounding_box))
    plt.xlim((-bounding_box, bounding_box))
    plt.show()

def random_batch(state_chunks, control_chunks, p_chunks):
    # Sample which chunk
    chunk_choices = np.random.choice(len(state_chunks), size=BATCH_SIZE, p=p_chunks)

    # Sample where to start in the chunk
    start_choices = [np.random.randint(len(state_chunks[i])) for i in chunk_choices]

    state_batch = []
    state_check_batch = []
    control_batch = []
    control_check_batch = []

    for chunk_choice in chunk_choices:
        start_choice = np.random.randint(len(state_chunks[chunk_choice]) - DOWNSAMPLE * (STATE_STEPS + CHECK_STEPS) + 1)

        state_batch.append(state_chunks[chunk_choice][start_choice:start_choice+DOWNSAMPLE * (STATE_STEPS + CHECK_STEPS):DOWNSAMPLE])
        control_batch.append(control_chunks[chunk_choice][start_choice:start_choice+DOWNSAMPLE * (STATE_STEPS + CHECK_STEPS):DOWNSAMPLE])

    state_batch = np.array(state_batch)
    control_batch = np.array(control_batch)

    state_check_batch = state_batch[:,-CHECK_STEPS:,:]
    control_check_batch = control_batch[:,-CHECK_STEPS:,:]
    state_batch = state_batch[:,:STATE_STEPS,:]
    control_batch = control_batch[:,:STATE_STEPS,:]

    return state_batch, control_batch, state_check_batch, control_check_batch

def normalize_batch(state_batch, state, name="normalize_batch"):
    with tf.variable_scope(name):
        state = tf.reshape(state[:,:3], (BATCH_SIZE, 1, 3))

        # Rotate the positions by the mean theta
        c, s = tf.cos(-state[:, :, 2]), tf.sin(-state[:, :, 2])
        R = tf.stack((
                tf.stack((c, -s), axis=2),
                tf.stack((s,  c), axis=2)),
                axis=2)
        R = tf.tile(R, (1, tf.shape(state_batch)[1], 1, 1))

        position = tf.matmul(R, tf.expand_dims(state_batch[:,:,:2] - state[:,:,:2], axis=-1))
        theta = state_batch[:,:,2] - state[:,:,2]

        state_batch = tf.concat((
            tf.reshape(position, (BATCH_SIZE, tf.shape(position)[1], 2)),
            tf.expand_dims(theta, axis=2),
            state_batch[:,:,3:5]),axis=2)

    return state_batch

def normalize_vector_batch(vector_batch, state, name="normalize_vector_batch"):
    c, s = tf.cos(-state[:, :, 2]), tf.sin(-state[:, :, 2])
    R = tf.stack((
            tf.stack((c, -s), axis=2),
            tf.stack((s,  c), axis=2)),
            axis=2)
    R = tf.tile(R, (1, tf.shape(vector_batch)[1], 1, 1))

    position = tf.matmul(R, tf.expand_dims(vector_batch[:,:,:2], axis=-1))
    vector_batch = tf.concat((
        tf.reshape(position, (BATCH_SIZE, tf.shape(position)[1], 2)),
        tf.expand_dims(vector_batch[:,:,2], axis=2),
        vector_batch[:,:,3:5]),axis=2)
    return vector_batch
 
def dense_net(input_, training, name="dense_net", reuse=False):
    """
    Make a dense neural net where each layer is an entry in
    layer_units. All but the last layer includes a nonlinearity.
    """
    hidden = input_

    with tf.variable_scope(name, reuse=reuse):

        for i, num_units in enumerate(LAYER_UNITS):
            activation = ACTIVATIONS[i]

            # Dense connection
            hidden = tf.layers.dense(
                    inputs=hidden,
                    units=num_units,
                    activation=activation,
                    kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV),
                    name="dense_" + str(i),
                    reuse=reuse)

            if not reuse:
                with tf.variable_scope("dense_"+str(i), reuse=True):
                    weights = tf.get_variable("kernel")
                    tf.summary.histogram("dense_" + str(i) + "_weights", weights)
                    bias = tf.get_variable("bias")
                    tf.summary.histogram("dense_" + str(i) + "_biases", bias)

            if i < len(LAYER_UNITS) - 1:
                # Batch renorm
                # https://arxiv.org/pdf/1702.03275.pdf
                hidden = tf.layers.batch_normalization(
                        hidden, 
                        training=training, 
                        name="bn_" + str(i), 
                        renorm=True,
                        fused=True,
                        reuse=reuse)

                # Dropout only if training
                dropout = tf.where(training, DROPOUT, 1)
                hidden = tf.nn.dropout(hidden, dropout)

    return hidden

def beta(x, name="beta"):
    with tf.variable_scope(name):
        return tf.concat((
                x,
                tf.square(tf.maximum(0.,x)),
                tf.square(tf.minimum(0.,x))),
                axis=2)

def quadratic_lag_model(state_batch, control_batch, reuse, name="quadratic_lag_model"):
    with tf.variable_scope(name, reuse=reuse):
        state_weights = tf.get_variable(
                name="state_weights", 
                initializer=tf.random_normal_initializer(stddev=STD_DEV),
                shape=(BATCH_SIZE, STATE_STEPS, STATES, 5 * 3))
        control_weights = tf.get_variable(
                name="control_weights", 
                initializer=tf.random_normal_initializer(stddev=STD_DEV),
                shape=(BATCH_SIZE, STATE_STEPS, STATES, 2 * 3))
        state_batch_beta = tf.expand_dims(beta(state_batch), axis=3)
        control_batch_beta = tf.expand_dims(beta(control_batch), axis=3)
        quadratic_lag = \
                tf.reduce_sum(tf.matmul(state_weights, state_batch_beta), axis=1) +\
                tf.reduce_sum(tf.matmul(control_weights, control_batch_beta), axis=1)
        quadratic_lag = tf.squeeze(quadratic_lag)

    return quadratic_lag

def f(state_batch, control_batch, training, reuse, name="f"):
    with tf.variable_scope(name):
        # Normalize
        origin_batch = tf.zeros((BATCH_SIZE, 1, 3))
        origin_batch = normalize_batch(origin_batch, state_batch[:, -1])
        state_batch = normalize_batch(state_batch, state_batch[:, -1])

        input_ = tf.concat((
            tf.layers.flatten(beta(state_batch)),
            tf.layers.flatten(beta(control_batch))),
            axis=1)

        output_ = dense_net(input_, training=training, reuse=reuse)
        output_ = tf.reshape(output_, (BATCH_SIZE, STATE_STEPS, STATES))

        # Unnormalize
        output_ = normalize_vector_batch(output_, origin_batch)

    return output_

def runge_kutta(i, h, state_batch, control_batch, control_check_batch, training, reuse, name="runge_kutta"):
    print(i, reuse)
    with tf.variable_scope(name):
        k1 = f(state_batch, control_batch, training, reuse)

        control_batch = tf.concat((control_batch[:,1:],tf.expand_dims(control_check_batch[:,i], axis=1)),axis=1)
        i += 1
        k2 = f(
            state_batch + k1 * h,
            control_batch,
            training, True)
        k3 = f(
            state_batch + k2 * h,
            control_batch,
            training, True)

        control_batch = tf.concat((control_batch[:,1:],tf.expand_dims(control_check_batch[:,i], axis=1)),axis=1)
        i += 1
        k4 = f(
            state_batch + k3 * 2 * h,
            control_batch,
            training, True)

    state_batch = state_batch + (h/3.) * (k1 + 2*k2 + 2*k3 + k4)
    return i, state_batch, control_batch

def compute_loss(h, state_batch, control_batch, state_check_batch, control_check_batch, training, reuse=False):
    i = 0
    loss = 0
    check = state_batch
    while i + 1 < CHECK_STEPS:
        check = tf.concat((check[:,2:], state_check_batch[:,i:i+2]), axis=1)
        if i > 0:
            reuse = True
        i, state_batch, control_batch = runge_kutta(i, h, state_batch, control_batch, control_check_batch, training, reuse)
        loss = loss + tf.reduce_sum(tf.square(state_batch - check))

    # Write for summaries
    differences = state_batch - check
    tf.summary.scalar("position_loss", tf.reduce_mean(POSITION_SCALING * tf.norm(differences[:,:,:2],axis=1)))
    tf.summary.scalar("theta_loss", tf.reduce_mean(THETA_SCALING * tf.abs(differences[:,:,2])))
    tf.summary.scalar("rpm_loss", tf.reduce_mean(RPM_SCALING * tf.abs(differences[:,:,3])))
    tf.summary.scalar("voltage_loss", tf.reduce_mean(VOLTAGE_SCALING * tf.abs(differences[:,:,4])))

    return loss

def main():
    # Read the input data
    t_chunks, state_chunks, control_chunks, p_chunks = read_chunks(TRAIN_DIR)
    t_chunks_val, state_chunks_val, control_chunks_val, p_chunks_val = read_chunks(VALIDATION_DIR)

    # Compute the average time step
    h = np.mean(np.diff(t_chunks[0]))

    # Make placeholders
    h_ph = tf.placeholder(tf.float32, shape=(), name="h")
    training_ph = tf.placeholder(tf.bool, name="training")
    state_batch_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, STATE_STEPS, 5), name="state_batch")
    control_batch_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, STATE_STEPS, 2), name="control_batch")
    state_check_batch_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, CHECK_STEPS, 5), name="state_check_batch")
    control_check_batch_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, CHECK_STEPS, 2), name="control_check_batch")

    # Compute the loss
    loss = compute_loss(h_ph, state_batch_ph, control_batch_ph, state_check_batch_ph, control_check_batch_ph, training_ph)

    tf.summary.scalar("loss", loss)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.polynomial_decay(
            LEARNING_RATE_START,
            global_step,
            LEARNING_RATE_END_STEPS,
            LEARNING_RATE_END,
            power=LEARNING_RATE_POWER)
    tf.summary.scalar("learning_rate", learning_rate)

    # Optimize
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # For batch norm
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    summary = tf.summary.merge_all()

    with tf.Session() as session:
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(LOG_DIR + "train", session.graph)
        validation_writer = tf.summary.FileWriter(LOG_DIR + "validation")
        baseline_writer = tf.summary.FileWriter(LOG_DIR + "baseline")

        saver = tf.train.Saver()

        for i in range(2000000):
            # Make random positive and unlabeled batches
            state_batch, control_batch, state_check_batch, control_check_batch = random_batch(
                    state_chunks, control_chunks, p_chunks)

            feed_dict = {}
            feed_dict[h_ph] = h
            feed_dict[state_batch_ph] = state_batch
            feed_dict[control_batch_ph] = control_batch
            feed_dict[state_check_batch_ph] = state_check_batch
            feed_dict[control_check_batch_ph] = control_check_batch
            feed_dict[training_ph] = True

            session.run(optimizer, feed_dict=feed_dict)

            if i % 100 == 0:
                train_summary = session.run(summary, feed_dict=feed_dict)

                feed_dict[h_ph] = 0
                feed_dict[training_ph] = False
                baseline_summary = session.run(summary, feed_dict=feed_dict)

                state_batch, control_batch, state_check_batch, control_check_batch = random_batch(
                        state_chunks_val, control_chunks_val, p_chunks_val)

                feed_dict[h_ph] = h
                feed_dict[state_batch_ph] = state_batch
                feed_dict[control_batch_ph] = control_batch
                feed_dict[state_check_batch_ph] = state_check_batch
                feed_dict[control_check_batch_ph] = control_check_batch

                validation_summary = session.run(summary, feed_dict=feed_dict)

                train_writer.add_summary(train_summary, i)
                validation_writer.add_summary(validation_summary, i)
                baseline_writer.add_summary(baseline_summary, i)
                print(i)

            if i % 100000 == 0:
                print("Saving...")
                saver.save(session, LOG_DIR + str(i) + "/model.ckpt")

if __name__ == "__main__":
    main()
