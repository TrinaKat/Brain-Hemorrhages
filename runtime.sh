#!/bin/bash

# Calculate runtime for each python ML script using the time() command
# Ignore program output other than accuracy

# TODO improve final results messages for scikit-learn

################################################################################
################################################################################
###############                      WARNING                    ################
###############         SCRIPT WILL TAKE FOREVER TO RUN         ################
###############                 OUTPUTS TO STDOUT               ################
################################################################################
################################################################################

# Tensorflow
echo -e "\n#####################\n     TENSORFLOW \n####################\n"

source ~/tensorflow/bin/activate

# Logistic Regression
echo -e " Logistic Regression \n---------------------"
time python ./TF_1.0.1/tfLogisticReg.py 2>/dev/null | grep "Final results*"
echo -e "\n"

# MLP Neural Network
echo -e " MLP Neural Network \n--------------------"
time python ./TF_1.0.1/tfNeuralNetwork.py 2>/dev/null | grep "Final results*"
echo -e "\n"

deactivate


# Scikit-Learn
echo -e "\n#####################\n    SCIKIT-LEARN \n#####################\n"

# Logistic Regression
echo -e " Logistic Regression \n---------------------"
time python ./skLogisticReg.py 2>/dev/null | grep -o "testing accuracy of.*"
echo -e "\n"

# MLP Neural Network
echo -e " MLP Neural Network \n--------------------"
time python ./skMLPClassifier.py 2>/dev/null | grep -o "testing accuracy of.*"
echo -e "\n"

# SVM
echo -e " Support Vector Machine (SVM) \n------------------------------"
time python ./SKL_0.18.1/skSVM.py 2>/dev/null
echo -e "\n"

# Random Forest
echo -e " Random Forest \n---------------"
time python ./SKL_0.18.1/skRandomForest.py 2>/dev/null | grep -o "testing accuracy of.*"
echo -e "\n"

# SGD Classifiers
echo -e " Linear Classifiers with SGD \n-----------------------------"
time python ./skSGDClassifier.py 2>/dev/null | grep -o "testing accuracy of.*"
echo -e "\n"

# Nearest Neighbors
echo -e " Nearest Neighbors \n-------------------"
time python ./skNearestNeighbors.py 2>/dev/null | grep -o "testing accuracy of.*"
echo -e "\n"

# Nearest Centroid
echo -e " Nearest Centroid \n------------------"
time python ./skNearestCentroid.py 2>/dev/null | grep -o "testing accuracy of.*"
echo -e "\n"

# Bagging Classifier
echo -e " Bagging Classifier \n--------------------"
time python ./skBaggingClassifier.py 2>/dev/null | grep -o "testing accuracy of.*"
echo -e "\n"

# Gradient Classifier
echo -e " Gradient Classifier \n---------------------"
time python ./skGradientClassifier.py 2>/dev/null | grep -o "testing accuracy of.*"

# DONE
echo -e "\n#####################\n ALL TESTS COMPLETED \n#####################\n"
