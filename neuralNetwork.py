""" 
Implementation of deep neural network to predict brain hemorrhaging for stroke patients
"""

import numpy as np
import tensorflow as tf 

# START INTERACTIVE SESSION

sess = tf.InteractiveSession()

# SET UP NODES FOR INPUTS AND OUTPUTS

# we are using a vector with 618 elements to represent each brain image's data
# "None" indicates that we will take in arbitrary vectors/rows of data 
X = tf.placeholder(tf.float32, shape=[None, 618])
# Boolean to indicate whether or brain actually hemorrhaged
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# SET UP WEIGHTS/PARAMETERS FOR MODEL

# initialize weights all to 0
W = tf.Variable(tf.zeros([618, 1]))
sess.run(tf.global_variables_initializer())

# IMPLEMENT MODEL

# currently, this is just a softmax regress model with a single linear layer
y = tf.matmul(X, W)

# TODO: train the model