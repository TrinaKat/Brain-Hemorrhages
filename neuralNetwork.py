""" 
Implementation of deep neural network to predict brain hemorrhaging for stroke patients
"""

# IMPORT REQUIRED LIBRARIES

import numpy as np
import tensorflow as tf

# OBTAIN DATA 

# read in training data (CSV file)
# 50000 rows, 623 columns
filePath = "../trainingData/train_data_final2.csv"
numRowsToSkip = 49000

# get all the column indices except the first four columns and the last column
columns = []
for i in range(4, 622):
	columns.append(i)

# get a 50000 x 618 column array for all of the values (just 1000 x 618 for now)
trainingData = np.loadtxt(filePath, delimiter = ',', skiprows = numRowsToSkip, usecols = tuple(columns))	
# get a 50000 x 1 column array for all of the results (boolean) (just 1000 x 1 for now)
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
correct_prediction = tf.equal(y, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TRAIN AND EVALUATE MODEL

# use gradient descent with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# run the training step on training set (800 samples)
for i in range(0, 800):
	# run one step of gradient descent
	sess.run(train_step, feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
	# check to see how accurate the model is so far
	if i % 50 == 0:
		train_accuracy = accuracy.eval(feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
		print("Training step %d: training accuracy %d"%(i, train_accuracy))


# run the evaluation on the test set (200 samples)
for i in range(800, 1000):
	test_accuracy = accuracy.eval(feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
	# check the test result accuracy 
	print("Testing step %d: testing accuracy %d"%(i, test_accuracy))