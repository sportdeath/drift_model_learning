#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

import time_stepping
import read_data
import process_data
import params

def dense_net(input_, training, layer_units=[1], activations=[None], name="dense_net", reuse=False):
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

        for i, num_units in enumerate(layer_units):
            # Perform the dense layer
            layer_name = "dense_" + str(i)
            hidden = tf.layers.dense(
                    inputs=hidden,
                    units=num_units,
                    activation=activations[i],
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

            if i + 1 < len(layer_units):
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

    with tf.variable_scope(name, reuse=reuse):
        # Normalize the states around the last pose
        state_batch_n = process_data.set_origin(state_batch, state_batch[:, -1])

        # Approximate the velocities using finite differences
        velocity_start_n = (state_batch_n[:,1,:] - state_batch_n[:,0,:])/h
        velocity_middle_n = (state_batch_n[:,2:,:] - state_batch_n[:,:-2,:])/(2 * h)

        # Combine the normalized states and controls
        # into one large state.
        input_ = tf.concat((
            tf.layers.flatten(state_batch_n),
            tf.layers.flatten(control_batch[:,:,params.THROTTLE_IND])),
            # tf.layers.flatten(control_batch[:,-1:,params.THROTTLE_IND])),
            axis=1)

        # Incorporate the steering command
        steer = dense_net(control_batch[:,:,params.STEER_IND], layer_units=[1], activations=[None], training=training, reuse=reuse, name="steer_net")
        # steer_scaling = tf.get_variable("steer_scaling", shape=[], dtype=tf.float32, initializer=tf.ones_initializer())
        # steer_bias = tf.get_variable("steer_bias", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer())
        # steer_bias = 0.
        # if not reuse:
            # tf.summary.scalar("steer_scaling", steer_scaling)
            # tf.summary.scalar("steer_bias", steer_bias)
        # steer = steer_scaling * control_batch[:,-1:,params.THROTTLE_IND] + steer_bias
        # steer = 1.05*control_batch[:,-1:,params.THROTTLE_IND]
        steer_components = tf.concat((tf.ones((tf.shape(steer)[0], 1)), tf.sin(steer), tf.cos(steer)), axis=1)

        # Rotate the input into the tire's frame of referece
        input_ = tf.layers.flatten(
                tf.tile(tf.expand_dims(steer_components, axis=1), (1, tf.shape(input_)[1], 1)) * \
                tf.tile(tf.expand_dims(input_, axis=2), (1, 1, tf.shape(steer_components)[1])))

        # Use a separate neural net to compute each state variable
        dx = dense_net(input_, layer_units=params.X_LAYER_UNITS, activations=params.X_ACTIVATIONS, training=training, reuse=reuse, name="ddx_net")
        dy = dense_net(input_, layer_units=params.Y_LAYER_UNITS, activations=params.Y_ACTIVATIONS, training=training, reuse=reuse, name="ddy_net")
        dtheta = dense_net(input_, layer_units=params.THETA_LAYER_UNITS, activations=params.THETA_ACTIVATIONS, training=training, reuse=reuse, name="ddtheta_net")

        # Change coordinate spaces again
        dx = tf.reduce_sum(steer_components * dx, axis=1)
        dy = tf.reduce_sum(steer_components * dy, axis=1)
        dtheta = tf.reduce_sum(steer_components * dtheta, axis=1)


        # Here is the normalized data
        dstate_batch_n = tf.stack((dx, dy, dtheta), axis=1)

        # Correct the end velocity using the learned model
        velocity_end_n_prev = velocity_middle_n[:,-1,:]
        velocity_end_n = velocity_end_n_prev + dstate_batch_n

        # We want the system to deaccelerate
        # |velocity_end_n| < |velocity_end_n_prev|
        stability_loss = tf.reduce_sum(tf.maximum(dstate_batch_n * (dstate_batch_n + 2 * velocity_end_n_prev), 0.))
        tf.add_to_collection("stability_losses", stability_loss)

        # Combine the slices
        velocity_n = tf.concat((
                tf.expand_dims(velocity_start_n, axis=1),
                velocity_middle_n,
                tf.expand_dims(velocity_end_n, axis=1)), axis=1)

        # Combine the results into one state
        # Un-normalize the data
        velocity = process_data.set_origin(velocity_n, -state_batch[:,-1], derivative=True)

    return velocity

def compute_loss(h, state_batch, control_batch, state_check_batch, control_check_batch, training, reuse=False):
    """
    Iterates a state using the dynamics functions and
    compares it to future states to determine the error.

    Args:
        h: The time step.
        state_batch: The input state tensor.
        control_batch: The input control tensor.
        state_check_batch: The future states to verify
        control_check_batch: The future controls to use as further input.
        training: A boolean tensor that is true iff we are training the network.
        reuse: True if we are reusing old network weights.
    """

    ts = time_stepping.RungeKutta(f)

    # Integrate
    i, next_state_batch, next_control_batch = ts.integrate(
            0, h, 
            state_batch, 
            control_batch, 
            control_check_batch, 
            training, 
            reuse)

    error = tf.abs(state_check_batch - next_state_batch[:,-params.CHECK_STEPS:])
    error_base = tf.abs(state_check_batch - tf.concat((state_batch, state_check_batch), axis=1)[:,-(params.CHECK_STEPS+1):-1])
    error_relative = error/(error_base + params.MIN_ERROR)
    error_loss = tf.reduce_sum(error_relative)

    stability_loss = tf.reduce_sum(tf.get_collection("stability_losses"))

    loss = error_loss + 0.01 * stability_loss

    # Write for summaries
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("stability_loss", stability_loss)
    tf.summary.scalar("error_loss", error_loss)
    tf.summary.scalar("x_loss", tf.reduce_mean(params.X_SCALING * error[:,:,params.X_IND]))
    tf.summary.scalar("y_loss", tf.reduce_mean(params.Y_SCALING * error[:,:,params.Y_IND]))
    tf.summary.scalar("theta_loss", tf.reduce_mean(params.THETA_SCALING * error[:,:,params.THETA_IND]))
    tf.summary.scalar("x_loss_rel", tf.reduce_mean(error_relative[:,:,params.X_IND]))
    tf.summary.scalar("y_loss_rel", tf.reduce_mean(error_relative[:,:,params.Y_IND]))
    tf.summary.scalar("theta_loss_rel", tf.reduce_mean(error_relative[:,:,params.THETA_IND]))

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
    learning_rate = tf.train.polynomial_decay(
            params.LEARNING_RATE,
            global_step,
            params.LEARNING_RATE_END_STEPS,
            params.LEARNING_RATE_END,
            power=params.LEARNING_RATE_POWER)
    tf.summary.scalar("learning_rate", learning_rate)

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

            if i % 1000 == 0:
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

            if i % 30000 == 0:
                print("Saving...")
                saver.save(session, params.LOG_DIR + str(i) + "/model.ckpt")
