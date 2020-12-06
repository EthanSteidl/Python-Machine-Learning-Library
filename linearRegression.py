import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

class LinearRegression():

    '''
    Constructor for class
    weights - (np.ndArray) holds weights for regression with first weight being the base
    threshold - (float) threshold for classifying true or false on model
    '''
    def __init__(self, threshold):
        self.weights = np.empty
        self.threshold = threshold

    '''
    Author: Ethan Steidl
    Will alter internal weights to produce a model that maps the input data X
    to the training data y. creating the hypotheses weights

    param X - (DataFrame) Input data
    param y - (DataFrame) Desired output

    return self - returns this object as a model since wieghts are stored
    '''
    def fit(self, X, y):

        # Equation from (Bremer) citation in Porfolio.PDF
        # formula B = (X'X)^-1 * X'y
        # B = Inverse( Transpose(X) * X ) * ( Transpose(X) * y )

        #create matrix X
        x_matrix = np.asmatrix(X)

        # insert the base 1 value into the X matrix
        x_matrix = np.insert(x_matrix, [0], [1], axis=1)

        #create matrix y
        y_matrix = np.asmatrix(y)

        #calculate (X'X)
        first = x_matrix.transpose().dot(x_matrix)

        #calculate (X'X)^-1 * X'
        second = np.linalg.inv(first).dot(x_matrix.transpose())

        #calculate (X'X)^-1 * X'y
        B = second.dot(y_matrix.transpose())

        #set weights
        self.weights = B
        return self


    '''
    Author: Ethan Steidl
    Given a Matrix X of vectors, ceates a temp vector the size of the amount 
    of vectors. For each vector in Matrix X, if the vector's values represent a pass,
    the corrosponding index in the temp vector is set to 1. Else it is set to 
    -1.  The temp vector is then returned.

    param X - (DataFrame) Input data

    return - Vector of length of X with 1's and -1's
    '''
    def predict(self,X):

        result = []

        #for each row in X find if the value is witin a threashold of the regression
        #if it is in the threshold apply a 1 value, if not apply -1
        for instance in X:

            value = self.weights[0].max()

            #for each row calculate its value from the linear equation of weights
            for index, feature in enumerate(instance):
                value += self.weights[index+1].max() * feature

            #add the absolute value of the result to the list since we are not concerned
            #about the sign
            result.append(abs(value))

        #find all values within the threshold and set them to 1 else -1
        return np.where(np.asarray(result) <= 1- self.threshold, 1, -1)


    '''
    Author: Ethan Steidl
    Plots the line of best fit through the data based on the weights.

    param X - (DataFrame) Input data
    param y - (DataFrame) classification data
    param resolution - (float) resolution of plot

    return void
    '''
    def plot(self, X, y, resolution=0.02):

        #place each point of the graph where blue points have an expected result
        #of 1 and red has an expected result of -1
        for item, answer in zip(X, y):
            c = 'red'
            if (answer == 1):
                c = 'blue'

            #plt.plot(item[0], item[1], 'x', color=c)
            plt.plot(item[0], 'x', color=c)

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

        xx1 = np.meshgrid(np.arange(x1_min, x1_max, resolution))

        plt.xlim(xx1[0].min(), xx1[0].max())


        x = np.linspace(start = x1_min, stop = x1_max)
        templist = self.weights.tolist()
        print(templist[0])
        plt.plot(x,  templist[0] + templist[1]*x)


        #graph
        plt.show()