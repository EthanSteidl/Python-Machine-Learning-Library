import numpy as np
import matplotlib.pyplot as plt
import matplotlib


'''
Author: Ethan Steidl
This implementation of the logistic regression is based of the online
article https://www.kdnuggets.com/2019/10/build-logistic-regression-model-python.html
The author of this artile is Abhinav Sagar.

This class is a binary classifier version of the logistic regression
it takes in data and trains similar to a perceptron. The learner
uses the sigmoid function to calculate its weights.
The exact equation can be found of the WIKI article linked below
https://en.wikipedia.org/wiki/Logistic_regression
After applying the sigmoid function, weights are altered to accomadate 
a better learner.

When performing a prediction, the learner runs the input data against
the sigmoid function to calculate probailites of sucess. Through testing
it was found when the sigmoid evaluates to a number greater than 
10 ^ -8 it classifies as a sucess or 1. Anything else is a failure
'''
class LogisticRegression(object):

    '''
    Author: Ethan Steidl
    weights (ndArray) - weights used for prediction
    iterations (int) - learning iteration count
    learning_rate (flaot) - how fast the model learns
    accuracy (float) - how accurate the model is for the training data
    '''
    def __init__(self, iterations = 10, learning_rate = 0.05):
        self.weights = np.empty
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.accuracy = 0.0

    '''
    Author: Ethan Steidl
    Calculates the logistic regression sigmoid function using a vector of weights.
    
    param scores - (ndArray) Array of feature weights
    
    return (ndArray) - predictions of the funtion with values 0.0 to 1.0
    '''
    def sigmoid(self,scores):
        return 1 / (1 + np.exp(-scores))

    '''
    Author: Ethan Steidl
    Performs the logistic regression learning algorithm.
    Weights are calculated through an iterative process where the features
    are doted against weights and their success probability is assessed by
    a sigmoid function. Then error is calculated and the weights are altered
    based on the diviation between the prediciton errors and the features.

    param X - (DataFrame) Input data
    param y - (DataFrame) Training Set
    iterations - (int) number of iterations
    learning_rate - (float) learning rate

    return (ndArray) - model weights
    '''
    def logistic_regression(self, X, y, iterations, learning_rate):

        #set weights to zero
        weights = np.zeros(X.shape[1])

        for step in range(iterations):
            #score is the features dotted against weights
            scores = np.dot(X, weights)

            #scores then processed thouugh sigmoid to calcualte probabilities
            predicted = self.sigmoid(scores)

            #from the probabilites, calculate the error and change rate
            prediction_error = y - predicted
            gradient = np.dot(X.transpose(), prediction_error)

            #adjust weights accordingly
            weights += learning_rate * gradient

        return weights


    '''
    Author: Ethan Steidl
    Fits the learning model and calculates the models accuracy

    param X - (DataFrame) Input data
    param y - (DataFrame) Training Set

    return void
    '''
    def fit(self, X, y):
        #calculate weights
        self.weights = self.logistic_regression(X, y, self.iterations, self.learning_rate)

        #calculate accuracy
        self.calculate_accuracy(X, y)


    '''
    Author: Ethan Steidl
    Calculates accuracy by comparing the output of the model against the
    expected ouput classifications

    param X - (DataFrame) Input data
    param y - (DataFrame) Training Set

    return void
    '''
    def calculate_accuracy(self, X, y):

        #the value 10^-8 was found to be the threshold for true through testing
        calculated = np.where(self.sigmoid(np.dot(X, self.weights)) > 10 ** -8, 1, -1)

        #finds the amount of correctly classified instances and calcualtes accuracy
        score = 0
        for i,j in zip(calculated,y):
            if (i == j):
                score += 1
        self.accuracy = score/y.shape[0]
        return

    '''
    Author: Ethan Steidl
    Performs predictions on input data. The predictions returned are a vector
    of -1 or 1 values. 1 represent success and -1 represents failure

    param X - (DataFrame) Input data

    return (ndArray) - Array of -1 and 1 result values
    '''
    def predict(self, X):
        # the value 10^-8 was found to be the threshold for true through testing
        return np.where(self.sigmoid(np.dot(X, self.weights)) > 10**-8, 1, -1)

