""" 
Implementation of deep neural network to predict brain hemorrhaging for stroke patients
"""

# IMPORT REQUIRED LIBRARIES

import numpy as np
import tensorflow as tf

# OBTAIN DATA 

# read in training data (CSV file)
filePath = "../trainingData/train_data_sm.csv"
numRowsToSkip = 39000

# get all the column indices except the first four columns and the last column
columns = []
for i in range(4, 622):
	columns.append(i)

# get a 40000 x 618 column array for all of the values (just 1000 x 618 for now)
trainingData = np.loadtxt(filePath, delimiter = ',', skiprows = numRowsToSkip, usecols = tuple(columns))	
# get a 40000 x 1 column array for all of the results (boolean) (just 1000 x 1 for now)
results = np.loadtxt(filePath, delimiter = ',', skiprows = numRowsToSkip, usecols = 622)

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

# get the predicted output from the model
# currently, this is just a softmax regression model with a single linear layer
y = tf.matmul(X, W)

# use the cross entropy as the error for the prediction
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# TRAIN AND EVALUATE MODEL

# use gradient descent with a learning rate of 0.5
# running the "train_step" operation will run one step of gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# run the training step on training set (800 samples)
for i in range(0, 800):
	sess.run(train_step, feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
	
# run the evaluation on the test set (200 samples)
# numCorrect = 0
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(800, 1000):
	result = sess.run(accuracy, feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
	if (i % 20 == 0):
		print result
# 	if (y[i] == y_):
# 		numCorrect += 1

# print "Accuracy of predictions on test set: {}%".format(numCorrect/200)