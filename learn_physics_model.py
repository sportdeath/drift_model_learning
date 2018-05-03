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
        cm_to_rear = 0.165
        cm_to_front = 0.165
        mass = 5.6
        inertia = 0.06
        # mass = 0.05

        theta = state_batch[:,2]
        rear_wheel_speed = state_batch[:,3]
        forward_rate = state_batch[:,4]
        lateral_rate = state_batch[:,5]
        yaw_rate = state_batch[:,6]
        steer = control_batch[:,0,params.STEER_IND]
        throttle = control_batch[:,0,params.THROTTLE_IND]

        # Compute the change in (x, y, theta)
        dx = forward_rate * tf.cos(theta) - lateral_rate * tf.sin(theta)
        dy = forward_rate * tf.sin(theta) + lateral_rate * tf.cos(theta)
        dtheta = yaw_rate

        lateral_rate_front = lateral_rate + cm_to_front * yaw_rate
        forward_rate_front = forward_rate
        lateral_rate_rear = lateral_rate - cm_to_rear * yaw_rate
        # Compute the force on the front tires themselves
        lateral_tire_rate_front = tf.cos(steer) * lateral_rate_front - tf.sin(steer) * forward_rate_front

        net_input = tf.stack((
            tf.cos(theta),
            tf.sin(theta),
            forward_rate,
            lateral_rate,
            yaw_rate,
            rear_wheel_speed,
            steer,
            lateral_rate_front,
            lateral_rate_rear,
            lateral_tire_rate_front,
            throttle,
            throttle - rear_wheel_speed
            ), axis=1)

        # Compute the lateral speeds at the front and rear of the vehicle

        lateral_tire_force_front = dense_net(net_input, training, reuse=reuse, name="lateral_tire_force_front")
        # lateral_force_front = dense_net(net_input, training, reuse=reuse, name="lateral_force_front")
        # forward_force_front = dense_net(net_input, training, reuse=reuse, name="forward_force_front")
        lateral_force_rear = dense_net(net_input, training, reuse=reuse, name="lateral_force_rear")
        forward_force_rear = dense_net(net_input, training, reuse=reuse, name="forward_force_rear")
        # dwheel_rate = dense_net(net_input, training, reuse=reuse, name="dwheel_rate")
        p_throttle = 3.
        dwheel_rate = p_throttle * (throttle - rear_wheel_speed)

        lateral_tire_force_front = tf.squeeze(lateral_tire_force_front)
        # lateral_force_front = tf.squeeze(lateral_force_front)
        # forward_force_front = tf.squeeze(forward_force_front)

        forward_force_rear = tf.squeeze(forward_force_rear)
        lateral_force_rear = tf.squeeze(lateral_force_rear)
        dwheel_rate = tf.squeeze(dwheel_rate)

        lateral_force_front = lateral_tire_force_front * tf.cos(steer)
        forward_force_front = lateral_tire_force_front * tf.sin(steer)

        dyaw_rate = (cm_to_front * lateral_force_front - cm_to_rear * lateral_force_rear)/inertia
        dforward_rate = lateral_rate * yaw_rate - (forward_force_front + forward_force_rear)/mass
        dlateral_rate = -forward_rate * yaw_rate + (lateral_force_front + lateral_force_rear)/mass

        dstate_batch = tf.stack((
            dx,
            dy,
            dtheta,
            dwheel_rate,
            dforward_rate,
            dlateral_rate,
            dyaw_rate), axis=1)

    return dstate_batch

def compute_loss(h, state_batch, control_batch, state_check_batch, control_check_batch, training, reuse=False):
    """
    """
    ts = time_stepping.RungeKutta(f)

    # Integrate
    i = 0
    next_state_batch = state_batch
    while i + 1 < params.CHECK_STEPS:
        if i > 0:
            reuse = True

        i, next_state_batch, next_control_batch = ts.integrate(
                i, h, 
                next_state_batch, 
                control_batch, 
                control_check_batch, 
                training, 
                reuse)

    error = state_check_batch[:,-1] - next_state_batch[:,:4]
    error_relative = error/(tf.abs(state_check_batch[:,-1] - state_batch[:,:4]) + params.MIN_ERROR)
    # loss = tf.reduce_sum(tf.square(error_relative))
    loss = tf.reduce_sum(tf.square(error))

    # Write for summaries
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("x_loss", tf.reduce_mean(params.X_SCALING * tf.abs(error[:,params.X_IND])))
    tf.summary.scalar("y_loss", tf.reduce_mean(params.Y_SCALING * tf.abs(error[:,params.Y_IND])))
    tf.summary.scalar("theta_loss", tf.reduce_mean(params.THETA_SCALING * tf.abs(error[:,params.THETA_IND])))
    tf.summary.scalar("rpm_loss", tf.reduce_mean(params.THROTTLE_SCALING * tf.abs(error[:,params.RPM_IND])))

    tf.summary.scalar("x_loss_rel", tf.reduce_mean(tf.abs(error[:,params.X_IND])))
    tf.summary.scalar("y_loss_rel", tf.reduce_mean(tf.abs(error[:,params.Y_IND])))
    tf.summary.scalar("theta_loss_rel", tf.reduce_mean(tf.abs(error[:,params.THETA_IND])))
    tf.summary.scalar("rpm_loss_rel", tf.reduce_mean(tf.abs(error[:,params.RPM_IND])))

    return loss

if __name__ == "__main__":
    # Read the input data
    t_chunks, state_chunks, control_chunks, p_chunks = read_data.read_chunks(params.TRAIN_DIR)
    t_chunks_val, state_chunks_val, control_chunks_val, p_chunks_val = read_data.read_chunks(params.VALIDATION_DIR)
    print(state_chunks[0].shape)

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

    # Guess the first state with 
    print(state_batch_ph)
    x = state_batch_ph[:,-1,params.X_IND]
    y = state_batch_ph[:,-1,params.Y_IND]
    theta = state_batch_ph[:,-1,params.THETA_IND]
    wheel_speed = state_batch_ph[:,-1,params.RPM_IND]

    state_batch_n = process_data.set_origin(state_batch_ph, state_batch_ph[:, -1])
    forward_rate = (state_batch_n[:,-1,params.X_IND] - state_batch_n[:,-2,params.Y_IND])/h
    lateral_rate = (state_batch_n[:,-1,params.Y_IND] - state_batch_n[:,-2,params.Y_IND])/h
    yaw_rate = (state_batch_n[:,-1,params.THETA_IND] - state_batch_n[:,-2,params.THETA_IND])/h

    input_state = tf.stack((
        x,
        y,
        theta,
        wheel_speed,
        forward_rate,
        lateral_rate,
        yaw_rate), axis=1)

    # Compute the loss
    loss = compute_loss(
            h_ph, 
            input_state,
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
