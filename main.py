import pandas as pd
import numpy as np
from ML import Perceptron
from ML import LinearRegression
from ML import AxisAlignedRectangle
from ML import KNN
from ML import DecisionStump
from ML import HardSVM
from ML import LogisticRegression
'''
This file contains example code for the
perceptron, axis aligned rectangle, and linear regression
'''

def axis_aligned_example():

    # captures data
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # divides up data, y is training set. X is input data
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1 , 1)

    X = df.iloc[0:100, [0, 3]].values

    #create an axis aligned rectangle
    aa = AxisAlignedRectangle()

    #fit the model
    aa.fit(X,y)

    #check dimensions, corners, and accuracy of the model
    print(aa.dimensions, aa.corners, aa.accuracy)

    #example of predicting values
    print(aa.predict(X))

    #plotting the graph
    aa.plot(X,y)

def linear_regression_example():
    # captures data
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # divides up data, y is training set. X is input data
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    #X = df.iloc[0:100, [0, 2]].values
    X = df.iloc[0:100, [0]].values

    # create a linearRegression object with threshold
    lr = LinearRegression(0.05)

    #fit the model
    lr.fit(X, y)

    #look at weights
    print(lr.weights)

    #predict some data set
    print(lr.predict(X))



def Hard_SVM_example():
    # captures data
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # divides up data, y is training set. X is input data
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    # create a perceptron object
    svm = HardSVM(0.1, 10)

    # fit the model
    svm.fit(X, y)

    # check the errors array
    print(svm.errors)

    # plot the graph
    svm.plot_decision_regions(X, y)


def perceptron_example():
    # captures data
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # divides up data, y is training set. X is input data
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    # create a perceptron object
    pn = Perceptron(0.1, 10)

    # fit the model
    pn.fit(X, y)

    # check the errors array
    print(pn.errors)

    # plot the graph
    pn.plot_decision_regions(X, y)

def KNN_example():
    # captures data
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # divides up data, y is training set. X is input data
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    #the prediction set is what the model will predict the classifier of
    predictionSet = df.iloc[45:59, [0, 2]].values

    #the plot value is what the model will try to find the nearest neighbors of
    #This is an arbitrary new point to run against the model
    plotValue = np.asarray([5.0, 2.5])

    # create a perceptron object
    knn = KNN()

    # fit the model
    knn.fit(X, y)

    #shows the predictions the model gave
    print(knn.predict(predictionSet, k=4))

    # plot the graph
    knn.plot(X, y, plotValue, k=5 )

def decision_stump_example():
    # captures data
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # divides up data, y is training set. X is input data
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    # create a perceptron object
    ds = DecisionStump()

    # fit the model
    ds.fit(X, y)

    #make prediction
    print(ds.predict(X))

    #check model accuracy
    print(ds.accuracy)

    # plot the graph
    ds.plot(X, y)

def logistic_regression_example():
    # captures data
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # divides up data, y is training set. X is input data
    y = df.iloc[30:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[30:100, [0, 2]].values

    # create a perceptron object
    lr = LogisticRegression()

    # fit the model
    lr.fit(X, y)

    # print the weights used
    print(lr.weights)

    #make prediction
    print(lr.predict(X))

    #check model accuracy
    print(lr.accuracy)




perceptron_example()
linear_regression_example()
logistic_regression_example()
axis_aligned_example()
decision_stump_example()
Hard_SVM_example()
KNN_example()