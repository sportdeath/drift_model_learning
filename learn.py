#!/usr/bin/env python3

"""
Train a neural network to learn
drift
"""

import os
import csv
import numpy as np
import tensorflow as tf

LOG_DIR = "tmp/drifter/new_data_9/"

"""
The number of units and the 
activation function used at the
output of each layer of the network
"""
LAYER_UNITS = [800, 5]
ACTIVATIONS = [tf.nn.relu, None]

"""
Percentage of the time that
a node is not dropped out.
"""
DROPOUT = 0.7

"""
The number of states to check
"""
STATE_STEPS = 10

"""
The number of future states to verify.
"""
CHECK_STEPS = 9

"""
The number of elements in a training batch.
"""
BATCH_SIZE = 20
LEARNING_RATE = 0.0002
THETA_SCALING = 1.
RPM_SCALING = 20000.
VOLTAGE_SCALING = 10.
STD_DEV = 0.001
TRAIN_DIR = "./train/"
VALIDATION_DIR = "./validation/"

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
    p_chunks = p_chunks/np.sum(p_chunks)

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
        start_choice = np.random.randint(len(state_chunks[chunk_choice]) - STATE_STEPS - CHECK_STEPS + 1)

        state_batch.append(state_chunks[chunk_choice][start_choice:start_choice+STATE_STEPS + CHECK_STEPS])
        control_batch.append(control_chunks[chunk_choice][start_choice:start_choice+STATE_STEPS + CHECK_STEPS])

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
                initializer=tf.zeros_initializer(),
                shape=(BATCH_SIZE, STATE_STEPS, 5, 5 * 3))
        control_weights = tf.get_variable(
                name="control_weights", 
                initializer=tf.zeros_initializer(),
                shape=(BATCH_SIZE, STATE_STEPS, 5, 2 * 3))
        state_batch_beta = tf.expand_dims(beta(state_batch), axis=3)
        control_batch_beta = tf.expand_dims(beta(control_batch), axis=3)
        quadratic_lag = \
                tf.reduce_sum(tf.matmul(state_weights, state_batch_beta), axis=1) +\
                tf.reduce_sum(tf.matmul(control_weights, control_batch_beta), axis=1)
        quadratic_lag = tf.squeeze(quadratic_lag)

    return quadratic_lag

def f(state_batch, control_batch, training, reuse, name="f"):
    with tf.variable_scope(name):
        quadratic_lag = quadratic_lag_model(state_batch, control_batch, reuse)

        input_ = tf.concat((
            tf.layers.flatten(beta(state_batch)),
            tf.layers.flatten(beta(control_batch))),
            axis=1)

        output_ = dense_net(input_, training=training, reuse=reuse)

    return quadratic_lag + output_

def forward_euler_loss(h, state_batch, control_batch, state_check_batch, control_check_batch, training, reuse=False, name="forward_euler_loss"):
    with tf.variable_scope(name):

        origin_batch = tf.zeros((BATCH_SIZE, 1, 3))
        for i in range(CHECK_STEPS):
            if i > 0:
                reuse = True

            origin_batch = normalize_batch(origin_batch, state_batch[:, -1])
            state_batch = normalize_batch(state_batch, state_batch[:, -1])

            predicted = state_batch[:,-1] + (state_batch[:,-1] - state_batch[:,-2] + h * f(state_batch, control_batch, training, reuse))

            # Combine the state with the previous ones
            state_batch = tf.concat((
                    state_batch[:,1:],
                    tf.expand_dims(predicted,axis=1)),
                    axis=1)
            control_batch = tf.concat((
                    control_batch[:,1:],
                    tf.expand_dims(control_check_batch[:,i], axis=1)),
                    axis=1)

        # Unnormalize the final state
        state_batch = normalize_batch(state_batch, origin_batch)

        loss = tf.reduce_sum(tf.square(state_batch[:,-1] - state_check_batch[:,-1]))

        differences = state_batch[:,-1] - state_check_batch[:,-1]
        tf.summary.scalar("position_loss", tf.reduce_mean(tf.norm(differences[:,:2],axis=1)))
        tf.summary.scalar("theta_loss", tf.reduce_mean(THETA_SCALING * tf.abs(differences[:,2])))
        tf.summary.scalar("rpm_loss", tf.reduce_mean(RPM_SCALING * tf.abs(differences[:,3])))
        tf.summary.scalar("voltage_loss", tf.reduce_mean(VOLTAGE_SCALING * tf.abs(differences[:,4])))

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
    loss = forward_euler_loss(h_ph, state_batch_ph, control_batch_ph, state_check_batch_ph, control_check_batch_ph, training_ph)

    tf.summary.scalar("loss", loss)
    summary = tf.summary.merge_all()

    # Optimize
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # For batch norm
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as session:
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(LOG_DIR + "train", session.graph)
        validation_writer = tf.summary.FileWriter(LOG_DIR + "validation")
        baseline_writer = tf.summary.FileWriter(LOG_DIR + "baseline")

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

if __name__ == "__main__":
    main()
