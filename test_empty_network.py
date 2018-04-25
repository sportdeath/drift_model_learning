import params
import plotting
import read_data
import process_data
import learn
import time_stepping
import tensorflow as tf

if __name__ == "__main__":
    # Read the data
    _, state_chunks, control_chunks, p_chunks = read_data.read_chunks(params.TRAIN_DIR)

    # Set the timestep low so we only see
    # the initialization effects
    # (it cannot be zero due to division)
    h = 0.0000000001

    # Make placeholders
    state_batch_ph = tf.placeholder(tf.float32, shape=(1, params.STATE_STEPS, params.STATES), name="state_batch")
    control_batch_ph = tf.placeholder(tf.float32, shape=(1, params.STATE_STEPS, params.CONTROLS), name="control_batch")
    state_check_batch_ph = tf.placeholder(tf.float32, shape=(1, params.CHECK_STEPS, params.STATES), name="state_check_batch")
    control_check_batch_ph = tf.placeholder(tf.float32, shape=(1, params.CHECK_STEPS, params.CONTROLS), name="control_check_batch")

    # Evaluate the next step
    ts = time_stepping.RungeKutta(learn.f)
    _, next_state_batch_tf, _ = ts.integrate(0, h, state_batch_ph, control_batch_ph, control_check_batch_ph, False, False)

    with tf.Session() as sess:

        # Initialize the network
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for i in range(3):
            # Get a random batch from the data
            state_batch, control_batch, state_check_batch, control_check_batch = process_data.random_batch(state_chunks, control_chunks, p_chunks)

            # Plot the batch and the check
            plotting.plot_states([state_batch, state_check_batch])

            # Evaluate the next state and plot it
            feed_dict = {}
            feed_dict[state_batch_ph] = state_batch[:1]
            feed_dict[control_batch_ph] = control_batch[:1]
            feed_dict[state_check_batch_ph] = state_check_batch[:1]
            feed_dict[control_check_batch_ph] = control_check_batch[:1]
            next_state_batch = sess.run(next_state_batch_tf, feed_dict=feed_dict)
            plotting.plot_states([state_batch, state_check_batch, next_state_batch])
