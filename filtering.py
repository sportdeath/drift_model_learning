import numpy as np

def gaussian(array, std_dev):
    normalization = 1/np.sqrt(2*np.pi*std_dev*std_dev)
    return normalization * np.exp(-(array*array)/(2*std_dev*std_dev))

def bilateral_filter(vector, num_neighbors, pos_dev, intensity_dev):
    """
    Args:
        vector: A 1D vector.
    """

    # Pad the vector with its neighbors by extending the ends
    vector_padded = np.pad(vector, (num_neighbors,), mode="edge")

    total_weight = np.zeros(vector.shape)
    vector_filtered = np.zeros(vector.shape)
    for i in range(2 *num_neighbors + 1):
        vector_shifted = vector_padded[i:len(vector)+i]

        pos_diff = num_neighbors - i
        intensity_diff = vector - vector_shifted

        weight = gaussian(pos_diff, pos_dev) * gaussian(intensity_diff, intensity_dev)
        total_weight += weight
        vector_filtered += vector_shifted * weight

    return vector_filtered/total_weight

if __name__ == "__main__":
    import read_data
    import params
    import plotting

    # Read the data
    t_chunks, state_chunks, control_chunks, p_chunks = read_data.read_chunks(params.TRAIN_DIR)

    # Plot the voltages and rpms
    voltages = []
    voltage = state_chunks[0][:,params.V_IND]
    voltages.append((0, voltage))
    voltages.append((0, bilateral_filter(voltage, 30, 6., 0.2)))
    # plotting.plot_vectors(voltages, title="Voltage")

    rpms = []
    rpm = state_chunks[0][:,params.RPM_IND]
    rpms.append((0, rpm))
    rpms.append((0, bilateral_filter(rpm, 30, 6., 0.2)))
    rpms.append((0, control_chunks[0][:,params.THROTTLE_IND]))
    plotting.plot_vectors(rpms, title="RPM")
