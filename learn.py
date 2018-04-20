"""
Train a neural network
"""

import csv
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

# Constants
# ACTIVATION = None
# ACTIVATION = tf.nn.leaky_relu
# tf.nn.relu
# LAYER_UNITS = [800, 800, 5]
# ACTIVATIONS = [None, tf.nn.relu, None]
LAYER_UNITS = [800, 5]
ACTIVATIONS = [tf.nn.relu, None]
DROPOUT = 0.7
STATE_STEPS = 10
CHECK_STEPS = 2
BATCH_SIZE = 10
LEARNING_RATE = 0.0002
TRAIN_PROPORTION = 0.7
LOG_DIR = "tmp/drifter/quadratic_lag_10_steps_quad_net_h1_thetad1_init_6/"
THETA_SCALING = 1.
RPM_SCALING = 20000.
VOLTAGE_SCALING = 10.
STD_DEV = 0.001

def parse_bag_file(bag_file):
    """
    Parses raw car data into numpy arrays.

    Args:
        One of Corey's .bag.csv files.

    Returns:
        t: A time vector.
        observed_states: 
    """

    with open(bag_file, 'r') as bag_file:
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

def random_batch(states, controls):
    """
    The inputs
    Returns:
        state_batch: [BATCH_SIZE, STATE_STEPS, 5]
        control_batch: [BATCH_SIZE, STATE_STEPS, 2]
        state_check_batch: [BATCH_SIZE, CHECK_STEPS, 5]
        control_check_batch: [BATCH_SIZE, CHECK_STEPS, 2]
    """

    # Uniformly sample the starting locations
    choices = np.random.randint(len(states) - STATE_STEPS - CHECK_STEPS + 1, size=(BATCH_SIZE, 1))

    # Sample the step size
    indices = np.expand_dims(np.arange(STATE_STEPS + CHECK_STEPS), axis=0)
    indices = np.tile(indices, (BATCH_SIZE, 1))
    choices = choices + indices

    state_batch = np.take(states, choices, axis=0)
    control_batch = np.take(controls, choices, axis=0)

    state_check_batch = state_batch[:,-CHECK_STEPS:,:]
    control_check_batch = control_batch[:,-CHECK_STEPS:,:]
    state_batch = state_batch[:,:STATE_STEPS,:]
    control_batch = control_batch[:,:STATE_STEPS,:]

    mean_pose_ = mean_pose(state_batch)
    state_batch = mean_offset(state_batch, mean_pose_)
    state_check_batch = mean_offset(state_check_batch, mean_pose_)

    return state_batch, control_batch, state_check_batch, control_check_batch

def mean_pose(state_batch):
    x_mean = np.mean(state_batch[:, :, 0], axis=1)
    y_mean = np.mean(state_batch[:, :, 1], axis=1)
    # theta_mean = np.arctan2(
            # np.sum(np.sin(state_batch[:, :, 2]), axis=1), 
            # np.sum(np.cos(state_batch[:, :, 2]), axis=1))
    theta_mean = np.mean(state_batch[:, :, 2], axis=1)

    mean = np.stack([x_mean, y_mean, theta_mean], axis=1)
    mean = np.reshape(mean, (-1, 1, 3))

    return mean

def mean_offset(state_batch, mean_pose_):
    state_batch[:,:,:3] -= mean_pose_

    # Rotate the positions by the mean theta
    c, s = np.cos(-mean_pose_[:, :, 2]), np.sin(-mean_pose_[:, :, 2])
    R = np.stack((
            np.stack((c, -s), axis=2),
            np.stack((s,  c), axis=2)),
            axis=2)

    state_batch = np.expand_dims(state_batch, axis=-1)
    state_batch[:, :, :2] = np.matmul(R, state_batch[:, :, :2])
    state_batch = np.squeeze(state_batch, axis=3)

    return state_batch
 
def dense_net(input_, training, name="dense_net", reuse=False):
    """
    Make a dense neural net where each layer is an entry in
    layer_units. All but the last layer includes a nonlinearity.
    """
    hidden = input_

    with tf.variable_scope(name, reuse=reuse):

        for i, num_units in enumerate(LAYER_UNITS):
            # Make the last layer linear
            # activation = None
            # if i < len(LAYER_UNITS) - 1:
                # activation = ACTIVATION 
            activation = ACTIVATIONS[i]

            # Dense connection
            hidden = tf.layers.dense(
                    inputs=hidden,
                    units=num_units,
                    activation=activation,
                    # kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                    kernel_initializer=tf.random_normal_initializer(stddev=STD_DEV),
                    # kernel_initializer=tf.zeros_initializer(),
                    # bias_initializer=tf.random_normal_initializer(),
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

def beta(x):
    s = tf.concat((
            x,
            tf.square(tf.maximum(0.,x)),
            tf.square(tf.minimum(0.,x))),
            axis=2)
    return s


def f(state_batch, control_batch, training):
    with tf.variable_scope("f"):
        with tf.variable_scope("quadratic_lag_model"):
            state_weights = tf.Variable(
                    tf.zeros((BATCH_SIZE, STATE_STEPS, 5, 5 * 3)),
                    name="state_weights", 
                    trainable=True)
            control_weights = tf.Variable(
                    tf.zeros((BATCH_SIZE, STATE_STEPS, 5, 2 * 3)),
                    name="control_weights", 
                    trainable=True)
            state_batch_beta = tf.expand_dims(beta(state_batch), axis=3)
            control_batch_beta = tf.expand_dims(beta(control_batch), axis=3)
            quadratic_lag = \
                    tf.reduce_sum(tf.matmul(state_weights, state_batch_beta), axis=1) +\
                    tf.reduce_sum(tf.matmul(control_weights, control_batch_beta), axis=1)
            quadratic_lag = tf.squeeze(quadratic_lag)

        input_ = tf.concat((
            tf.layers.flatten(beta(state_batch)),
            tf.layers.flatten(beta(control_batch))),
            axis=1)

        output_ = dense_net(input_, training=training)

    return quadratic_lag + output_
    # return quadratic_lag

# def runge_kutta_loss(state_batch, control_batch, state_check_batch, control_check_batch):
    # k1 = f(state_batch, control_batch)
    # k2 = f(state_batch + h * k1, tf.concat((control_batch[:,1:,:], control_check_batch[:,:1,:]), axis=1))
    # k3 = f(state_batch + h * k2, tf.concat((control_batch[:,1:,:], control_check_batch[:,:1,:]), axis=1))
    # k4 = f(state_batch + 2 * h * k3, tf.concat((control_batch[:,2:,:], control_check_batch[:,:2,:]), axis=1))
    # control_check_batch[:,1

def forward_euler_loss(h, state_batch, control_batch, state_check_batch, control_check_batch, training):
    predicted = state_batch[:,-1] + (state_batch[:,-1] - state_batch[:,-2] + h * f(state_batch, control_batch, training))
    # predicted = state_batch[:,-1] + h * f(state_batch, control_batch, training)

    loss = tf.reduce_sum(tf.square(predicted - state_check_batch[:,0]))

    differences = predicted - state_check_batch[:,0]
    tf.summary.scalar("position_loss", tf.reduce_mean(tf.norm(differences[:,:2],axis=1)))
    tf.summary.scalar("theta_loss", tf.reduce_mean(THETA_SCALING * tf.abs(differences[:,2])))
    tf.summary.scalar("rpm_loss", tf.reduce_mean(RPM_SCALING * tf.abs(differences[:,3])))
    tf.summary.scalar("voltage_loss", tf.reduce_mean(VOLTAGE_SCALING * tf.abs(differences[:,4])))

    return loss

def main():
    # Read the input data
    t, states, controls = parse_bag_file("random_driving_chunk_4.bag.csv")
    num_train = int(TRAIN_PROPORTION * len(states))
    states_train = states[:num_train]
    controls_train = controls[:num_train]
    states_validation = states[num_train:]
    controls_validation = controls[num_train:]

    # Compute the average time step
    h = np.mean(np.diff(t))
    # h = 1.

    # Make placeholders
    h_ph = tf.placeholder(tf.float32, name="h")
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
                    states_train, controls_train)

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
                        states_validation, controls_validation)

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
