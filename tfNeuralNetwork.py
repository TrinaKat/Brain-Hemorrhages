""" 
Implementation of multilayer perceptron (deep neural network) to predict brain hemorrhaging for stroke patients
"""

# IMPORT REQUIRED LIBRARIES

import numpy as np
import tensorflow as tf

# SET PARAMETERS

# parameters for training/testing
learningRate = 0.5  # for gradient descent
numTrainingExamples = 30000
numTestingExamples = 20000
totalNumExamples = numTrainingExamples + numTestingExamples

# parameters for the neural network
n_hidden_1 = 64 # number of features in 1st hidden layer
n_hidden_2 = 32 # number of features in 2nd hidden layer

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

# IMPLEMENT MODEL

# define model
def multilayer_perceptron(X, weights, biases):
    # hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([618, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([1]))
}

# instantiate model
prediction = multilayer_perceptron(X, weights, biases)

# define cost function and optimizer for each training step
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, targets=y_))
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# used for testing the data
predicted_class = tf.greater(tf.sigmoid(prediction), 0.5)
correct_prediction = tf.equal(predicted_class, tf.equal(y_, 1.0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # variable for whether prediction is correct (0 or 1)

# initialize the global variables
sess.run(tf.global_variables_initializer())

# TRAIN MODEL

# run the training step on training set 
numCorrectTrainingExamples = 0
for i in range(0, numTrainingExamples):
	# run one step of gradient descent
	sess.run(train_step, feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
	# check to see how accurate the model is so far
	train_accuracy = accuracy.eval(feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
	numCorrectTrainingExamples += train_accuracy
	if i % 20 == 0 and i != 0:
		print("Training step %d: training accuracy %f%%"%(i, (numCorrectTrainingExamples/i) * 100))

# TEST MODEL

# run the evaluation on the test set 
numCorrectTestExamples = 0;
for i in range(numTrainingExamples, totalNumExamples):
	test_accuracy = accuracy.eval(feed_dict={X: np.reshape(trainingData[i], (1, 618)), y_: np.reshape(results[i], (1, 1))})
	numCorrectTestExamples += test_accuracy
	# check the test result accuracy 
	if i % 20 == 0 and i != numTrainingExamples:
		print("Testing step %d: testing accuracy %f%%"%(i, (numCorrectTestExamples/(i - numTrainingExamples)) * 100))

# usually around 70% for training set and 50% for test set
print("Final results: training accuracy of %f%%, testing accuracy of %f%%"%((numCorrectTrainingExamples/numTrainingExamples) * 100, (numCorrectTestExamples/numTestingExamples) * 100))