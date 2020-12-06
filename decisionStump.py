import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
'''
Author: Ethan Steidl
The Decision Stump class utilizes the decision stump macine learning
model. This model is explained in great detail at
https://en.wikipedia.org/wiki/Decision_stump#:~:text=A%20decision%20stump%20is%20a,just%20a%20single%20input%20feature.
The model selects a single instance of data with specific feature to base
its predictions off of. When predicting data, there is a single if statement
that decides the classifiacation. The training is done by comparing each element in
a feature column against all other elements in that column. That elements score is
the number of expected results guessed correctly where correct results have that
feature greater than the value being tested. This is done for each feature column
and the feature that provides the most accurate results is selected for use.
'''
class DecisionStump(object):

    '''
    Author: Ethan Steidl
    Constructor for DecisionStump class

    col (Int) - Index of feature to use for comparison
    value (Any) - Value to compare against in Stump
    accuracy (float) - How accuratly the model fits the training data
    greater_than (Bool) - Whether or not the model is flipped
            flipping the model can increase accuracy if the model guesses
            with very low accuracy. This changes the accuracy to 1-accuracy
    '''
    def __init__(self):
        self.col = 0
        self.value = 0
        self.accuracy = 0.0
        self.greater_than = True


    '''
    Author: Ethan Steidl
    Takes input data and returns a vector of classifications
    
    param X (DataFrame) - input data
    
    return (ndArray) - vector of -1 or 1 classification values
    '''
    def predict(self, X):

        #select specific column from the data for decision
        selected_column = X[:,[self.col]]

        #compare against the stump and return the result
        #greater_than will flip the result and increase accuracy
        if(self.greater_than == True):
            return np.where(selected_column >= self.value, 1, -1)
        else:
            return np.where(selected_column < self.value, -1, 1)


    '''
    Author: Ethan Steidl
    Fits the learning model by finding a single feature that classifies the most
    instances in the training set correctly.
    
    param X (DataFrame) - input data
    param y (DataFrame) - classification results
    
    return void
    '''
    def fit(self, X, y):

        #assume y is a 1d vector of 0 or 1 values

        Xflipped = X.transpose()


        best_value = 0
        best_count = 0
        best_index = 0
        best_greater_than = True
        max = X.shape[0]

        index = 0
        #for each col in original data
        for row in Xflipped:

            best_col_value = 0
            best_col_count = 0
            best_col_index = 0
            best_col_greater_than = True

            #for each element in that col
            for value in row:

                count = 0
                #compare the value against everything else in the col
                for item, result in zip(row,y):
                    if( (value > item and result == 1) or (value < item and result == -1)):
                        count += 1

                #optimization step, if the accuracy is bad, flip how comparisons occure
                if(count < max/2):
                    count = max-count
                    greater_than = False
                else:
                    greater_than = True

                #check to see if the row used is the best so far
                if(count > best_col_count ):
                    best_col_count = count
                    best_col_greater_than = greater_than
                    best_col_value = value

            #take the best column result calculated
            if(best_col_count > best_count):
                best_count = best_col_count
                best_value = best_col_value
                best_index = index
                best_greater_than = best_col_greater_than

            index += 1

        #set values to those of the best result
        self.col = best_index
        self.value = best_value
        self.accuracy = best_count / max
        self.greater_than = best_greater_than

    '''
    Author: Ethan Steidl
    Plots the learning model on a graph

    param X (DataFrame) - input data
    param y (DataFrame) - classification results
    param resolution (float) - resolution of graph

    return void
    '''
    def plot(self, X, y, resolution=0.02):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))

        #if the feature value is in column 1 use a horizontal fill
        if(self.col == 1):
            plt.hlines(self.value, xx1.min(), xx1.max(), color = 'black')

            plt.fill_between(np.asarray([xx1.min(), xx1.max()]), np.asarray([self.value, self.value]), 0,
                             facecolor="red",  # The fill color
                             color='red',  # The outline color
                             alpha=0.2)  # Transparency of the fill
            plt.fill_between(np.asarray([xx1.min(), xx1.max()]), np.asarray([self.value, self.value]), xx2.max(),
                             facecolor="blue",  # The fill color
                             color='blue',  # The outline color
                             alpha=0.2)  # Transparency of the fill

        # if the feature value is in column 1 use a vertical fill
        if(self.col == 0):
            plt.vlines(self.value, xx2.min(), xx2.max(), color='black')

            plt.fill_between(np.asarray([xx2.min(), xx2.max()]), np.asarray([self.value, self.value]), 0,
                             facecolor="red",  # The fill color
                             color='red',  # The outline color
                             alpha=0.2)  # Transparency of the fill
            plt.fill_between(np.asarray([xx2.min(), xx2.max()]), np.asarray([self.value, self.value]), xx1.max(),
                             facecolor="blue",  # The fill color
                             color='blue',  # The outline color
                             alpha=0.2)  # Transparency of the fill

        #min and max of plot
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=np.array([cmap(idx)]),
                        marker=markers[idx], label=cl)

        plt.show()