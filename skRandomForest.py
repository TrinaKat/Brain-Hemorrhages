"""
Implementation of decision trees and random forests to predict brain hemorrhaging for stroke patients
"""

# IMPORT REQUIRED LIBRARIES

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier


# SET PARAMETERS

numTrainingExamples = 30000
numTestingExamples = 20000
totalNumExamples = numTrainingExamples + numTestingExamples
n_estimators = 30  # number of trees in the forest

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
# standard normally distributed data: Gaussian with zero mean and unit variance
trainingData_scaled = preprocessing.scale(trainingData)

# get a 50000 x 1 column array for all of the results (boolean) (just 1000 x 1 for now)
results = np.loadtxt(filePath, delimiter = ',', skiprows = numRowsToSkip, usecols = 622)

# TRAIN THE MODELS

# accuracy: decision trees (92%), random forest (95%), extra-trees (96%), AdaBoost with decision trees (92%)
models = [DecisionTreeClassifier(max_depth=None), RandomForestClassifier(n_estimators=n_estimators), ExtraTreesClassifier(n_estimators=n_estimators), AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=n_estimators)]

# randomly split the data into training set and test set (40% testing)
X_train, X_test, y_train, y_test = train_test_split(trainingData_scaled, results, test_size=0.4, random_state=0)

# train and evaluate all of the models
for model in models:
	# train the model
	model.fit(X_train, y_train)

	# evaluate the trained model on the test set
	testAccuracy = model.score(X_test, y_test)

	print("Final results for '%s': testing accuracy of %f%%"%(model, testAccuracy * 100))