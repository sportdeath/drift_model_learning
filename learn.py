#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

import time_stepping
import read_data
import process_data
import params

def dense_net(input_, training, name="dense_net", reuse=False):
    """
    Regress the input using a fully connected neural network.
    The network parameters are detailed in the params file.

    Args:
        input_: The input tensor to the network.
        training: A boolean tensor that is true if the netork
            is being trained.
        reuse: If true, the network is being reused and the
            weights will be shared with previous copies.
        name: The name of the operation.

    Returns:
        The output of the neural network.
    """
    hidden = input_

    with tf.variable_scope(name, reuse=reuse):

        for i, num_units in enumerate(params.LAYER_UNITS):
            # Perform the dense layer
            layer_name = "dense_" + str(i)
            hidden = tf.layers.dense(
                    inputs=hidden,
                    units=num_units,
                    activation=params.ACTIVATIONS[i],
                    kernel_initializer=params.KERNEL_INITIALIZER,
                    name=layer_name,
                    reuse=reuse)

            if not reuse:
                # Add the histograms for debugging
                with tf.variable_scope(layer_name, reuse=True):
                    weights = tf.get_variable("kernel")
                    tf.summary.histogram("weights", weights)
                    bias = tf.get_variable("bias")
                    tf.summary.histogram("biases", bias)

            if i + 1 < len(params.LAYER_UNITS):
                if params.BATCH_NORM:
                    # Batch renorm
                    # https://arxiv.org/pdf/1702.03275.pdf
                    hidden = tf.layers.batch_normalization(
                            hidden, 
                            training=training, 
                            name="bn_" + str(i), 
                            renorm=True,
                            fused=True,
                            reuse=reuse)

                if params.DROPOUT:
                    # Dropout only if training
                    keep_prob = tf.where(training, params.KEEP_PROB, 1)
                    hidden = tf.nn.dropout(hidden, keep_prob)

    return hidden

def feature_expansion(x, name="feature_augmentation"):
    """
    Expands the features of a tensor by combining an
    input tensor with functions of itself.

    Args:
        x: The input tensor.
        name: The name of the operation.

    Returns:
        The expanded tensor.
    """

    with tf.variable_scope(name):
        x_expanded = tf.concat((
                x,
                x*x,
                tf.sin(x),
                tf.cos(x),
                tf.atan(x)),
                axis=-1)

        return x_expanded

def f(h, state_batch, control_batch, training, reuse, name="f"):
    """
    Compute the derivative at a state given its control input.
    The function is computed using multiple neural net regressions
    which can be trained.

    Args:
        h: The constant timestep.
        state_batch: The states.
        control_batch: The control inputs.
        training: A boolean tensor that is true if the network
            is being trained.
        reuse: If true, the network is being reused and the
            weights will be shared with previous copies.
        name: The name of the operation.
    """

    with tf.variable_scope(name):
        # Normalize the states around the last pose
        state_batch_n = process_data.set_origin(state_batch, state_batch[:, -1])

        # Combine the normalized states and controls
        # into one large state.
        input_ = tf.concat((
            tf.layers.flatten(state_batch_n),
            tf.layers.flatten(control_batch)),
            axis=1)

        # Augment the features with useful functions
        # input_ = feature_expansion(input_)

        # Use a separate neural net to compute each state variable
        x = dense_net(input_, training=training, reuse=reuse, name="x_net")
        y = dense_net(input_, training=training, reuse=reuse, name="y_net")
        theta = dense_net(input_, training=training, reuse=reuse, name="theta_net")

        # Here is the normalized data
        dstate_batch_n = tf.stack((x, y, theta), axis=2)

        # Combine the results into one state
        # Un-normalize the data
        dstate_batch = process_data.set_origin(dstate_batch_n, -state_batch[:,-1], derivative=True)

        # Approximate the velocities using finite differences
        velocity_start = (state_batch[:,1,:] - state_batch[:,0,:])/h
        velocity_middle = (state_batch[:,2:,:] - state_batch[:,:-2,:])/(2 * h)
        velocity_end = (state_batch[:,-1,:] - state_batch[:,-2,:])/h

        # Correct the end velocity using the learned model
        # velocity_end = dstate_batch + velocity_middle[:,-1:]
        velocity_end = dstate_batch + params.DECAY * tf.expand_dims(velocity_end, axis=1)

        # Combine the slices
        velocity = tf.concat((
                tf.expand_dims(velocity_start, axis=1),
                velocity_middle,
                velocity_end), axis=1)

    return velocity

def compute_loss(h, state_batch, control_batch, state_check_batch, control_check_batch, training, reuse=False):
    """
    """
    ts = time_stepping.RungeKutta(f)
    # ts = time_stepping.ForwardEuler(f)

    # Integrate
    i, next_state_batch, next_control_batch = ts.integrate(
            0, h, 
            state_batch, 
            control_batch, 
            control_check_batch, 
            training, 
            reuse)

    error = state_check_batch[:,-1] - next_state_batch[:,-1]
    error_relative = error/(tf.abs(state_check_batch[:,-1] - state_batch[:,-1]) + params.MIN_ERROR)
    loss = tf.reduce_sum(tf.square(error_relative))
    # loss = tf.reduce_sum(tf.abs(error))

    # Write for summaries
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("x_loss", tf.reduce_mean(params.X_SCALING * tf.abs(error[:,params.X_IND])))
    tf.summary.scalar("y_loss", tf.reduce_mean(params.X_SCALING * tf.abs(error[:,params.Y_IND])))
    tf.summary.scalar("theta_loss", tf.reduce_mean(params.THETA_SCALING * tf.abs(error[:,params.THETA_IND])))
    tf.summary.scalar("x_loss_rel", tf.reduce_mean(tf.abs(error_relative[:,params.X_IND])))
    tf.summary.scalar("y_loss_rel", tf.reduce_mean(tf.abs(error_relative[:,params.Y_IND])))
    tf.summary.scalar("theta_loss_rel", tf.reduce_mean(tf.abs(error_relative[:,params.THETA_IND])))

    return loss

if __name__ == "__main__":
    # Read the input data
    t_chunks, state_chunks, control_chunks, p_chunks = read_data.read_chunks(params.TRAIN_DIR)
    t_chunks_val, state_chunks_val, control_chunks_val, p_chunks_val = read_data.read_chunks(params.VALIDATION_DIR)

    # Compute the average time step
    h = np.mean(np.diff(t_chunks[0]))

    # Make placeholders
    h_ph = tf.placeholder(tf.float32, shape=(), name="h")
    training_ph = tf.placeholder(tf.bool, name="training")
    state_batch_ph = tf.placeholder(
            tf.float32, 
            shape=(params.BATCH_SIZE, params.STATE_STEPS, params.STATES), 
            name="state_batch")
    control_batch_ph = tf.placeholder(
            tf.float32, 
            shape=(params.BATCH_SIZE, params.STATE_STEPS, params.CONTROLS), 
            name="control_batch")
    state_check_batch_ph = tf.placeholder(
            tf.float32, 
            shape=(params.BATCH_SIZE, params.CHECK_STEPS, params.STATES), 
            name="state_check_batch")
    control_check_batch_ph = tf.placeholder(
            tf.float32, 
            shape=(params.BATCH_SIZE, params.CHECK_STEPS, params.CONTROLS), 
            name="control_check_batch")

    # Compute the loss
    loss = compute_loss(
            h_ph, 
            state_batch_ph, 
            control_batch_ph, 
            state_check_batch_ph, 
            control_check_batch_ph, 
            training_ph)

    # Initialize the learning rate
    global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.polynomial_decay(
            # params.LEARNING_RATE,
            # global_step,
            # LEARNING_RATE_END_STEPS,
            # LEARNING_RATE_END,
            # power=LEARNING_RATE_POWER)
    # tf.summary.scalar("learning_rate", learning_rate)

    # Minimize the loss function
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # For batch norm
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(params.LEARNING_RATE).minimize(loss, global_step=global_step)

    with tf.Session() as session:
        # Initialize the session
        summary = tf.summary.merge_all()
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())

        # Create writers for tensorboard
        train_writer = tf.summary.FileWriter(params.LOG_DIR + "train", session.graph)
        validation_writer = tf.summary.FileWriter(params.LOG_DIR + "validation")
        baseline_writer = tf.summary.FileWriter(params.LOG_DIR + "baseline")
        saver = tf.train.Saver()

        for i in range(2000000):
            # Make a random batch
            state_batch, control_batch, state_check_batch, control_check_batch = process_data.random_batch(
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

                feed_dict[h_ph] = 0.000000001
                feed_dict[training_ph] = False
                baseline_summary = session.run(summary, feed_dict=feed_dict)

                state_batch, control_batch, state_check_batch, control_check_batch = process_data.random_batch(
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

            if i % 10000 == 0:
                print("Saving...")
                saver.save(session, params.LOG_DIR + str(i) + "/model.ckpt")
