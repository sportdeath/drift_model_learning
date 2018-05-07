import os
import csv
import numpy as np

import params

def read_chunks(directory):
    """
    Read the RACECAR data.

    Args:
        The directory containing the data.

    Returns:
        Time, state, and control data. The data comes as a list of
        "chunks" where each chunk is a numpy array of the data from
        a particular test run.

        A probability array is also returned, representing the relative
        weighting of each chunk of data.

        t_chunks: The time data.
        state_chunks: The state data.
        control_chunks: The control data.
        p_chunks: An array whose values sum to 1 where the ith value is 
            proportional to the length of the ith chunk of data.
    """
    # Initialize zero data
    t_chunks = []
    state_chunks = []
    control_chunks = []

    for filename in os.listdir(directory):
        if filename.endswith(params.DATA_EXT):
            f = os.path.join(directory, filename)
            t, state, control = read_chunk(f)

            t_chunks.append(t)
            state_chunks.append(state)
            control_chunks.append(control)

    # Write a pro
    p_chunks = np.array([len(chunk) for chunk in state_chunks])
    p_chunks = p_chunks/float(np.sum(p_chunks))

    return t_chunks, state_chunks, control_chunks, p_chunks


def read_chunk(file_path):
    """
    Read a chunk of data.

    Args:
        file_path: The relative file path

    Returns:
    """

    with open(file_path, 'r') as bag_file:
        # Read the data from file
        reader = csv.reader(bag_file)
        data = []
        for row in reader:
            if row[0][0] == '#':
                continue
            data.append(row)

        # Convert the data to a numpy array
        data = np.array(data, dtype=np.float64)

        # Extract t, state, control
        t = data[:, 0]
        t -= t[0]
        controls = data[:, 1:3]
        states = data[:, 5:8]

        # Change theta = 0 to point in the positive x direction
        # and unwrap the angles
        # states[:, params.THETA_IND] += np.pi/2.
        states[:, params.THETA_IND] = np.unwrap(states[:, 2])

        # Normalize the steering angle around 0
        controls[:, params.STEER_IND] -= 0.52

        # Scale all the states for normalization
        states[:, params.X_IND] /= params.X_SCALING
        states[:, params.Y_IND] /= params.Y_SCALING
        states[:, params.THETA_IND] /= params.THETA_SCALING
        controls[:, params.THROTTLE_IND] /= params.THROTTLE_SCALING
        controls[:, params.STEER_IND] /= params.STEER_SCALING

    return t, states, controls

if __name__ == "__main__":
    """
    Test the file reading.
    """
    import plotting

    ind = 2
    t, state, control, p = read_chunks(params.TRAIN_DIR)
    plotting.plot_vectors([(0, control[ind][:,params.THROTTLE_IND]), (0, state[ind][:,params.X_IND]), (0, state[ind][:,params.Y_IND])],title="Throttle and position")
    plotting.plot_vectors([(0, control[ind][:,params.STEER_IND]), (0, state[ind][:,params.THETA_IND])],title="Steer and theta")
