"""
Implementation of binary logistic regression to predict brain hemorrhaging for stroke patients
"""

# IMPORT REQUIRED LIBRARIES

import numpy as np
import tensorflow as tf

# SET PARAMETERS

learningRate = 0.5  # for gradient descent
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

# START INTERACTIVE SESSION

sess = tf.InteractiveSession()

# SET UP NODES FOR INPUTS AND OUTPUTS

# we are using a vector with 618 elements to represent each brain image's data
# "None" indicates that we will take in arbitrary vectors/rows of data
X = tf.placeholder(tf.float32, shape=[None, 618])
# Boolean to indicate whether or brain actually hemorrhaged
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# SET UP WEIGHTS/PARAMETERS FOR MODEL

# initialize weights and bias all to 0
W = tf.Variable(tf.zeros([618, 1]))
b = tf.Variable(tf.zeros([1, 1]))
sess.run(tf.global_variables_initializer())

# IMPLEMENT MODEL

# get the predicted output from the model
# currently, this is just a logistic regression model with a single linear layer
y = tf.matmul(X, W) + b

# use the cross entropy as the error for the prediction (sigmoid since binary logistic regression)
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

# used for testing the data
prediction = tf.sigmoid(y)  # float between 0 and 1 that represents probability of hemorrhage
predicted_class = tf.greater(prediction, 0.5)
correct_prediction = tf.equal(predicted_class, tf.equal(y_, 1.0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # variable for whether prediction is correct (0 or 1)

# TRAIN MODEL

# use gradient descent with specified learning rate
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(cross_entropy)
# run the training step on training set
numCorrectTrainingExamples = 0
for i in range(0, numTrainingExamples):
	# run one step of gradient descent
	sess.run(train_step, feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
	# check to see how accurate the model is so far
	train_accuracy = accuracy.eval(feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
	numCorrectTrainingExamples += train_accuracy
	# if i % 20 == 0 and i != 0:
	if i % 1000 == 0 and i != 0:
		print("Training step %d: training accuracy %f%%"%(i, (numCorrectTrainingExamples/i) * 100))

# TEST MODEL

# run the evaluation on the test set
numCorrectTestExamples = 0;
for i in range(numTrainingExamples, totalNumExamples):
	test_accuracy = accuracy.eval(feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
	numCorrectTestExamples += test_accuracy
	# check the test result accuracy
	# if i % 20 == 0 and i != numTrainingExamples:
	if i % 1000 == 0 and i != numTrainingExamples:
		print("Testing step %d: testing accuracy %f%%"%(i, (numCorrectTestExamples/(i - numTrainingExamples)) * 100))

# usually around 96% for training set and 80% for test set
print("Final results: training accuracy of %f%%, testing accuracy of %f%%"%((numCorrectTrainingExamples/numTrainingExamples) * 100, (numCorrectTestExamples/numTestingExamples) * 100))
