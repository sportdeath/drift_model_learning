import numpy as np
import tensorflow as tf

import params

def random_flips(state_batch, control_batch, flips=[1, -1]):
    flip = np.random.choice(flips, size=state_batch.shape[0])
    flip = np.reshape(flip, (-1, 1))
    state_batch_ = np.copy(state_batch)
    control_batch_ = np.copy(control_batch)
    state_batch_[:,:,params.Y_IND] *= flip
    state_batch_[:,:,params.THETA_IND] *= flip
    control_batch_[:,:,params.STEER_IND] *= flip

    return state_batch_, control_batch_

def random_batch(state_chunks, control_chunks, p_chunks):
    """
    Generate random batches of data from the dataset.

    Args:
        state_chunks: A list of state arrays.
        control_chunks: A list of control input arrays.
        p_chunks: The weighting of each chunk

    Returns:
        state_batch:
        control_batch:
        state_check_batch: A batch with CHECK_STATES states.
    """
    # Sample which chunk
    chunk_choices = np.random.choice(len(state_chunks), size=params.BATCH_SIZE, p=p_chunks)

    # Sample where to start in the chunk
    start_choices = [np.random.randint(len(state_chunks[i])) for i in chunk_choices]

    state_batch = []
    state_check_batch = []
    control_batch = []
    control_check_batch = []

    sample_length = params.DOWNSAMPLE * (params.STATE_STEPS + params.CHECK_STEPS)

    for chunk_choice in chunk_choices:
        # Choose where to start the sample
        start_choice = np.random.randint(len(state_chunks[chunk_choice]) - sample_length + 1)

        # Sample the state and controls there
        state_batch.append(state_chunks[chunk_choice][start_choice: start_choice + sample_length :params.DOWNSAMPLE])
        control_batch.append(control_chunks[chunk_choice][start_choice: start_choice + sample_length :params.DOWNSAMPLE])

    # Make the state and controls into a numpy array
    state_batch = np.array(state_batch)
    control_batch = np.array(control_batch)

    state_batch, control_batch = random_flips(state_batch, control_batch)

    # Divide the arrays into input and verification sections
    state_check_batch = state_batch[:,-params.CHECK_STEPS:,:]
    control_check_batch = control_batch[:,-params.CHECK_STEPS:,:]
    state_batch = state_batch[:,:params.STATE_STEPS,:]
    control_batch = control_batch[:,:params.STATE_STEPS,:]

    return state_batch, control_batch, state_check_batch, control_check_batch


def set_origin(state_batch, origin_state, derivative=False, name="normalize_batch"):
    """
    Normalizes a batch of states around a new origin state.

    Args:
        state_batch: A batch of states with shape=(?, ?, STATES)
        origin_state: A state with shape=(?, STATES).
        derivative: If true, the state_batch is assumed to be the
            derivative, and only the rotation is accounted for.
        name: The name of the operation.

    Returns:
        state_batch: The normalized array.
    """

    with tf.variable_scope(name):
        # Rotate the positions by the mean theta
        c, s = tf.cos(-origin_state[:, params.THETA_IND]), tf.sin(-origin_state[:, params.THETA_IND])
        R = tf.stack((
                tf.stack((c, -s), axis=1),
                tf.stack((s,  c), axis=1)),
                axis=1)
        R = tf.expand_dims(R, axis=1)
        R = tf.tile(R, (1, tf.shape(state_batch)[1], 1, 1))

        x = state_batch[:,:,params.X_IND]
        y = state_batch[:,:,params.Y_IND]
        theta = state_batch[:,:,params.THETA_IND]

        if not derivative:
            # Center the states around origin
            x = x - tf.reshape(origin_state[:,params.X_IND], (-1, 1))
            y = y - tf.reshape(origin_state[:,params.Y_IND], (-1, 1))
            theta = theta - tf.reshape(origin_state[:,params.THETA_IND], (-1, 1))

        # Rotate the positions so theta is zero
        position = tf.matmul(R, tf.expand_dims(tf.stack((x, y), axis=2), axis=3))

        state_batch = tf.concat((
            position[:,:,:,0],
            tf.expand_dims(theta, axis=2)),
            axis=2)

    return state_batch

if __name__ == "__main__":
    import plotting
    import read_data

    # Read the data
    _, state_chunks, control_chunks, p_chunks = read_data.read_chunks(params.TRAIN_DIR)

    with tf.Session().as_default():
        for i in range(3):

            # Get a random batch from the data
            state_batch, control_batch, state_check_batch, control_check_batch = random_batch(state_chunks, control_chunks, p_chunks)

            # Plot the batch and the check
            print("Original.")
            plotting.plot_states([state_batch, state_check_batch])

            state_batch_, control_batch_ = random_flips(state_batch, control_batch, flips=[-1])
            state_check_batch_, control_check_batch_  = random_flips(state_check_batch, control_check_batch, flips=[-1])
            print("Flipping!")

            plotting.plot_states([state_batch_, state_check_batch_])

            # Plot that same vector normalized
            state_batch_n_tf = set_origin(state_batch, state_batch[:,-1])
            state_check_batch_n_tf = set_origin(state_check_batch, state_batch[:,-1])
            # plotting.plot_states([state_batch_n_tf.eval(), state_check_batch_n_tf.eval()])

            # Take the differences in the normalized frame
            diff_n_tf = state_batch_n_tf[:,1:] - state_batch_n_tf[:,:-1]
            # Return them to the original frame
            diff_tf = set_origin(diff_n_tf, -state_batch[:,-1], derivative=True)
            # Add them back to the original vector and plot the results
            state_batch_pred_tf = state_batch[:,:-1] + diff_tf
            plotting.plot_states([state_batch, state_check_batch, state_batch_pred_tf.eval()])

            # Plot the batch and the check again for verification
            plotting.plot_states([state_batch, state_check_batch])
