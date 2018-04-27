import numpy as np
import tensorflow as tf

"""
The log directory for tensorboard.
"""
LOG_DIR = "tmp/drifter2/first1_forward_l2_c10_rela_cumu_larger_error_filt/"

"""
The number of input states to the neural network.
"""
STATE_STEPS = 15

"""
The number of future states to verify in training.
"""
CHECK_STEPS = 10

"""
The number of units and the activation functions
used at the output of each layer of the network.
"""
LAYER_UNITS = [500, 1]
ACTIVATIONS = [tf.nn.relu, None]
# LAYER_UNITS = [1]
# ACTIVATIONS = [None]

"""
The initializer in the neural network.
"""
INIT_STD_DEV = 0.001
# INIT_STD_DEV = 0.1
KERNEL_INITIALIZER = tf.random_normal_initializer(stddev=INIT_STD_DEV)
# KERNEL_INITIALIZER = tf.contrib.layers.xavier_initializer()

"""
"""
MIN_ERROR = np.array([0.001, 0.001, 0.001, 0.1, 0.1])

"""
"""
INIT_WITH_FINITE_DIFFERENCES = True

"""
Should we use batch normalization?
"""
BATCH_NORM = False

"""
The integer factor to downsample the data.
"""
DOWNSAMPLE = 1

"""
The probability that a node is not dropped out.
"""
DROPOUT = False
KEEP_PROP = 0.7

"""
The number of elements in a training batch.
"""
BATCH_SIZE = 10

"""
"""

"""
The learning rate of the neural network.
"""
LEARNING_RATE = 0.0001
# LEARNING_RATE = 0.1

"""
The number of states in the system.
"""
STATES = 5

"""
The number of control inputs to the system.
"""
CONTROLS = 2

"""
The indices of each state variable.
"""
X_IND = 0
Y_IND = 1
THETA_IND = 2
RPM_IND = 3
V_IND = 4

"""
The indices of each control command.
"""
THROTTLE_IND = 0
STEER_IND = 1

"""
Factors to scale each variable before entering the network.
"""
X_SCALING = 1.
Y_SCALING = 1.
THETA_SCALING = 1.
RPM_SCALING = 20000.
V_SCALING = 10.
THROTTLE_SCALING = 20000.
STEER_SCALING = 1.

"""
The directories of training and validation data.
"""
TRAIN_DIR = "./train/"
VALIDATION_DIR = "./validation/"

"""
The file extension for training data.
"""
DATA_EXT = ".bag.csv"
