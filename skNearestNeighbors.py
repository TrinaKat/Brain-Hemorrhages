# #!/usr/bin/python

"""
Implementation of nearest neighbors algorithms to predict brain hemorrhaging for stroke patients
"""

# IMPORT REQUIRED LIBRARIES

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier


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
# since SVMs aren't scale invariant, need to scale data
# standard normally distributed data: Gaussian with zero mean and unit variance
trainingData_scaled = preprocessing.scale(trainingData)

# get a 50000 x 1 column array for all of the results (boolean) (just 1000 x 1 for now)
results = np.loadtxt(filePath, delimiter = ',', skiprows = numRowsToSkip, usecols = (622,))

# TRAIN THE MODELS

# randomly split the data into training set and test set (40% testing)
X_train, X_test, y_train, y_test = train_test_split(trainingData_scaled, results, test_size=0.4, random_state=0)

### K Neighbors Regressor: 80% ###

model = KNeighborsRegressor(n_neighbors=5)
# Fit the model according to the given training data
model.fit(X_train, y_train)

# evaluate the trained model on the test set
# Returns the mean accuracy on the given test data and labels
testAccuracy = model.score(X_test, y_test)

print("Final results for '%s': testing accuracy of %f%%"%(model, testAccuracy * 100))

### K Neighbors Classifier: 93% ###

model = KNeighborsClassifier(n_neighbors=5, weights='distance')
# Fit the model according to the given training data
model.fit(X_train, y_train)

# evaluate the trained model on the test set
# Returns the mean accuracy on the given test data and labels
testAccuracy = model.score(X_test, y_test)

print("Final results for '%s': testing accuracy of %f%%"%(model, testAccuracy * 100))

### Radius Neighbors Classifier: 85.5% ###

model = RadiusNeighborsClassifier(radius=8, outlier_label=0, weights='uniform')
# Fit the model according to the given training data
model.fit(X_train, y_train)

# evaluate the trained model on the test set
# Returns the mean accuracy on the given test data and labels
testAccuracy = model.score(X_test, y_test)

print("Final results for '%s': testing accuracy of %f%%"%(model, testAccuracy * 100))

