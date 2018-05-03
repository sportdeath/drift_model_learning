#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

import time_stepping
import read_data
import process_data
import params

def f(h, state_batch, control_batch, training, reuse, name="f"):
    """
    Compute the derivative at a state given its control input.
    This is based on the tire slip model
    http://planning.cs.uiuc.edu/node695.html

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
        # Normalize the data into the current frame
        # state_batch_n = process_data.set_origin(state_batch, state_batch[:, -1])

        # Compute the change in (x, y, theta)
        dx = forward_rate * tf.cos(theta) - lateral_rate * tf.sin(theta)
        dy = forward_rate * tf.sin(theta) + lateral_rate * tf.cos(theta)
        dtheta = yaw_rate

        # Initialize the distances from the center of mass to the 
        # front and rear axles of the vehicle
        cm_to_front = tf.Variable()
        cm_to_rear = tf.Variable()

        # The friction coefficient at the front tires
        cornering_stiffness_front = tf.Variable()
        # The friction coefficient at the rear tires
        cornering_stiffness_rear = tf.Variable()

        # Compute the lateral speeds at the front and rear of the vehicle
        lateral_rate_front = lateral_rate + cm_to_front * yaw_rate
        forward_rate_front = forward_rate
        lateral_rate_rear = lateral_rate - cm_to_rear * yaw_rate
        # Compute the force on the front tires themselves
        lateral_rate_frontwheel = tf.cos(steer) * lateral_rate_front - tf.sin(steer) * forward_rate_front

        # Convert those lateral speeds to forces
        lateral_force_front = -cornering_stiffness_front * FUNCTION(lateral_rate_frontwheel)
        lateral_force_rear = -cornering_stiffness_rear * FUNCTION(lateral_rate_rear)

        # Compute the accelerations
        dyaw_rate = (cm_to_front * force_front * tf.cos(steer) - cm_to_back * force_back)/intertia
        dforward_rate = lateral_rate * yaw_rate - (lateral_force_front * tf.sin(steer))/mass + FUNCTION2(thrust - forward_rate)
        dlateral_rate = -forward_rate * yaw_rate + (lateral_force_front * tf.cos(steer) + lateral_force_rear)/mass

        lateral_tire_force_front
        forward_force_rear
        lateral_force_rear

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
