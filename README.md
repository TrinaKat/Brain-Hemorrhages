# BrainHemorrhage
by Katrina Wijaya and Karen Li

## Purpose
Predicting brain hemorrhages from patient MRI scans

## Background
This is a term project for UCLA CS 188: Computational Methods for Medical Imaging taught by Professor Fabien Scalzo.

For more information about our project, check out our [website](https://predictingbrainhemorrhages.wordpress.com/)!

## Data
Data for this project comes from 263 acute ischemic stroke patients treated in four different medical centers. It is in the form of a pre-processed CSV file that has been manually processed by Professor Scalzo's team. The data will not be released as it contains privileged information of patients.

## Frameworks and Algorithms
Our main frameworks are TensorFlow and Scikit-Learn. If TensorFlow was used, script name begins with "tf", else "sk". We are exploring different frameworks and machine learning algorithms to see how well each performs in predicting brain hemorrhages from the pre-processed brain scans.

Using TensorFlow, we implemented a basic model with logistic regression and a neural network with a multilayer perceptron model.

Using Scikit-Learn, we implemented many classifier models such as nearest neighbors, SVM, and trees among other models. In addition, we also implemented logistic regression and a MLP neural network to enable comparison between the TensorFlow and Scikit-Learn libraries.

## Installation
Download or clone the BrainHemorrhage repository. You would need access to the data, but the data is not public so either include your own data and set the file path in the scripts or contact Professor Scalzo for more information. You need to have TensorFlow and Scikit-learn installed.

The scripts include the following libraries:  
numpy  
tensorflow  
sklearn 

The TensorFlow Python API supports Python 2.7 and Python 3.3+.

Scikit-learn requires:  
Python (>= 2.6 or >= 3.3),  
NumPy (>= 1.6.1),  
SciPy (>= 0.9).  

Depending on what version of Scikit-learn and TensorFlow you have, there may be errors with the scripts. The included directories SKL_0.18.1 and TF_1.0.1 have a minor syntax change that works for that version. You can check what version of TensorFlow and Scikit-learn you have with the version.py script. 

## Usage

In terminal in the BrainHemorrhage repository, enter the following command-line command to run the desired script:

```
$ python scriptToRun.py
```
