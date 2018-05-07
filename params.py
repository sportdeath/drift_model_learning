import numpy as np
import tensorflow as tf

"""
The log directory for tensorboard.
"""
LOG_DIR = "tmp/drift/29-steer-net/"

"""
The number of input states to the neural network.
"""
STATE_STEPS = 10

"""
The number of future states to verify in training.
"""
CHECK_STEPS = 2

"""
The number of basis vectors to be multiplied with
each steer component.
"""
# NUM_STEER_COMPONENTS = 1 + 2 * STATE_STEPS
NUM_STEER_COMPONENTS = 3

"""
The number of units and the activation functions
used at the output of each layer of the network.
"""
X_LAYER_UNITS = [20, NUM_STEER_COMPONENTS]
Y_LAYER_UNITS = [20, NUM_STEER_COMPONENTS]
THETA_LAYER_UNITS = [20, NUM_STEER_COMPONENTS]
X_ACTIVATIONS = [tf.nn.tanh, None]
Y_ACTIVATIONS = [tf.nn.tanh, None]
THETA_ACTIVATIONS = [tf.tanh, None]

"""
The initializer in the neural network.
"""
KERNEL_INITIALIZER = tf.contrib.layers.xavier_initializer()

"""
The minimum relative error considered.
"""
MIN_ERROR = 0.001

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
BATCH_SIZE = 100

"""
The learning rate of the neural network.
"""
LEARNING_RATE = 0.0001
LEARNING_RATE_END_STEPS = 2000000
LEARNING_RATE_END = 0.00001
LEARNING_RATE_POWER = 1

"""
The number of states in the system.
"""
STATES = 3

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
