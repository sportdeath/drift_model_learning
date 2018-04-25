import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.colors

import params

def plot_vector(t):
    """
    Plot a list of values with equal spacing.

    Args:
        t: A list of values.
    """
    s = np.arange(len(t))
    plt.figure()
    plt.title('Vector')
    plt.plot(s, t)
    plt.show()

def plot_states(states, bounding_box=None):
    """
    Plots several state arrays. Each array will be
    given a different color.

    Args:
        states: A list of numpy arrays, each of which
            are a batch of states. Only the first batch
            element will be plotted.
        bounding_box: The limits of the plot in each
            direction from the origin. If unspecified,
            the plot will fit the input data.
    """

    plt.figure()
    plt.title('States')
    plt.axes().set_aspect('equal')
    if bounding_box is not None:
        plt.ylim((-bounding_box, bounding_box))
        plt.xlim((-bounding_box, bounding_box))

    colors = cycle(('xkcd:black', 'xkcd:blue', 'xkcd:red', 'xkcd:violet', 'xkcd:green', 'xkcd:goldenrod'))

    for state_batch in states:
        x = state_batch[0,:,params.X_IND]
        y = state_batch[0,:,params.Y_IND]
        u = np.cos(state_batch[0,:,params.THETA_IND])
        v = np.sin(state_batch[0,:,params.THETA_IND])

        plt.quiver(x, y, u, v, scale=40., headwidth=3., width=0.002, color=next(colors))

    plt.show()

if __name__ == "__main__":
    """
    Test basic plotting.
    """
    import read_data

    # Read the data
    t_chunks, state_chunks, control_chunks, p_chunks = read_data.read_chunks(params.TRAIN_DIR)

    # Plot the time
    plot_vector(t_chunks[0])

    # Plot a couple states
    states = []
    for i in range(4):
        states.append(np.expand_dims(state_chunks[i], axis=0))
    plot_states(states)

    # Plot the states zoomed in
    states = []
    chunk = np.expand_dims(state_chunks[0], axis=0)
    for i in range(3):
        states.append(chunk[:,i*params.STATE_STEPS:(i+1)*params.STATE_STEPS])
    plot_states(states)