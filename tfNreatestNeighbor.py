"""
Implementation of nearest neighbor algorithm to predict brain hemorrhaging for stroke patients
"""
'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf

# SET PARAMETERS

numTrainingExamples = 30000
numTestingExamples = 20000
totalNumExamples = numTrainingExamples + numTestingExamples

# OBTAIN DATA

# read in training data (CSV file)
# 50000 rows, 623 columns
filePath = "../trainingData/train_data_final2.csv"
numRowsToSkip = 50000 - totalNumExamples

# get all the column indices except the first four columns and the last column
columns = []
for i in range(4, 622):
    columns.append(i)

# get a 50000 x 618 column array for all of the values (just 1000 x 618 for now)
trainingData = np.loadtxt(filePath, delimiter = ',', skiprows = numRowsToSkip, usecols = tuple(columns))
# get a 50000 x 1 column array for all of the results (boolean) (just 1000 x 1 for now)
results = np.loadtxt(filePath, delimiter = ',', skiprows = numRowsToSkip, usecols = 622)

# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# # In this example, we limit mnist data
# Xtr, Ytr = trainingData.train.next_batch(numTrainingExamples) #5000 for training (nn candidates)
# Xte, Yte = trainingData.test.next_batch(numTestingExamples) #200 for testing

# SET UP NODES FOR INPUTS AND OUTPUTS

# we are using a vector with 618 elements to represent each brain image's data
# "None" indicates that we will take in arbitrary vectors/rows of data
xtr = tf.placeholder(tf.float32, shape=[None, 618])
# Boolean to indicate whether or brain actually hemorrhaged
xte = tf.placeholder(tf.float32, shape=[None, 1])

# # tf Graph Input
# xtr = tf.placeholder("float", [None, 618])
# xte = tf.placeholder("float", [618])

# # SET UP WEIGHTS/PARAMETERS FOR MODEL

# # initialize weights and bias all to 0
# W = tf.Variable(tf.zeros([618, 1]))
# b = tf.Variable(tf.zeros([1, 1]))

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# used for testing the data
# prediction = tf.sigmoid(pred)  # float between 0 and 1 that represents probability of hemorrhage
predicted_class = tf.equal(pred, 1)
correct_prediction = tf.equal(predicted_class, tf.equal(xte, 1.0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # variable for whether prediction is correct (0 or 1)


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
numCorrectTrainingExamples = 0
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(0, numTrainingExamples):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: np.reshape(trainingData[i], (1,618)), xte: np.reshape(results[i], (1,1))})
        # # Get nearest neighbor class label and compare it to its true label
        # print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
        #     "True Class:", np.argmax(Yte[i]))
        # # Calculate accuracy
        # if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
        #     accuracy += 1./len(Xte)
        train_accuracy = accuracy.eval(feed_dict={xtr: np.reshape(trainingData[i], (1, 618)), xte: np.reshape(results[i], (1, 1))})
        numCorrectTrainingExamples += train_accuracy
        # if i % 20 == 0 and i != 0:
        if i % 1000 == 0 and i != 0:
            print("Training step %d: training accuracy %f%%"%(i, (numCorrectTrainingExamples/i) * 100))
    # TEST MODEL

    # run the evaluation on the test set
    numCorrectTestExamples = 0;
    for i in range(numTrainingExamples, totalNumExamples):
        test_accuracy = accuracy.eval(feed_dict={xtr: np.reshape(trainingData[i], (1, 618)), xte: np.reshape(results[i], (1, 1))})
        numCorrectTestExamples += test_accuracy
        # check the test result accuracy
        # if i % 20 == 0 and i != numTrainingExamples:
        if i % 1000 == 0 and i != numTrainingExamples:
            print("Testing step %d: testing accuracy %f%%"%(i, (numCorrectTestExamples/(i - numTrainingExamples)) * 100))

    # pretty sure I effed this up
    # lol it's guessing randomly (50% lollll)
    # what am I doing
    # must read more tutorials and cry more
    print("Final results: training accuracy of %f%%, testing accuracy of %f%%"%((numCorrectTrainingExamples/numTrainingExamples) * 100, (numCorrectTestExamples/numTestingExamples) * 100))

    print("Done!")
