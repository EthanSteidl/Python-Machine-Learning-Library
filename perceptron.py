import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches


class Perceptron(object):
    '''
    Constructor for class
    rate - (float) learning rate
    niter - (int) number of iterations
    weights - (Array[float]) weights for perceptron. Weights are 1 larger than Features in
            hypothoses class. Index 0 represents the bias.
    errors - (Array[int]) number of errors per itteration. On fit convergence last elemnt will
            be 0
    '''

    def __init__(self, rate=0.01, niter=10):
        self.rate = rate
        self.niter = niter  # number of itterations
        self.weights = np.empty
        self.errors = []
        self.features = 0

    '''
    Will alter internal weights to produce a model that maps the input data X
    to the training data y. creating the hypotheses weights

    param X - (DataFrame) Input data
    param y - (DataFrame) Desired output

    return self - returns this object as a model since wieghts are stored
    '''

    def fit(self, X, y):
        # clear erros array from previous fit if there was one
        self.errors.clear()
        # x and y are both df objects
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]"""
        # weights: create a weights array of right size
        # and initialize elements to zero

        self.features = X.shape[1]
        self.weights = [0 for x in range(X.shape[1] + 1)]  # 1 more weight for bias than features ie cols

        # main loop to fit the data to the labels
        for i in range(self.niter):
            # set iteration error to zero
            misclass = 0

            # loop over all the objects in X and corresponding y element
            for xi, target in zip(X, y):
                # calculate the needed (delta_w) update from previous step
                # delta_w = rate * (target – prediction current object)
                delta_w = self.rate * (target - self.predict(xi))

                # calculate what the current object will add to the weight
                self.weights[1:] = self.weights[1:] + (delta_w * xi)

                # set the bias to be the current delta_w
                self.weights[0] = + delta_w

                # increase the iteration error if delta_w != 0
                if (delta_w != 0):
                    misclass += 1

            # Number of misclassifications, creates an array
            # to hold the number of misclassifications
            # Update the misclassification array with # of errors in iteration


            self.errors.append(misclass)
            # if there was convergence, stop looping
            if (misclass == 0):
                break

        return self

    '''
    Author: Christer Karlsson
    Returns the dot product of the input data and the wights with the bias
    added to each elemnt.

    param X - (DataFrame) Input data

    return z - the result of X.Weights + bias
    '''

    def net_input(self, X):
        """Calculate net input"""
        # return the return the dot product: X.w + bias


        try:
            z = X.dot(self.weights[1:]) + self.weights[0]
        except:
            z = np.array([0 for x in self.features])

        return z

    '''
    Author: Christer Karlsson
    Given a Matrix X of vectors, ceates a temp vector the size of the amount 
    of vectors. For each vector in Matrix X, if the vector's values represent a pass,
    the corrosponding index in the temp vector is set to 1. Else it is set to 
    -1.  The temp vector is then returned.

    param X - (DataFrame) Input data

    return - Vector of length of X with 1's and -1's
    '''

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    '''
    Author: Christer Karlsson
    This function will plot the input data X and color the differing types different
    colors.  A line will be drawn across the graph where the perceptron classifies each
    element as a pass or fail.  Passes are in the red highlighted area and fails are in the
    blue highlighted area

    Added small change to remove error. when calling plt.scatter, the color needs to be an np
    array.

    param X - (DataFrame) Input data
    param y - (DataFrame) Training Set
    param resolution - (flot) resolution on graph

    return void
    '''

    def plot_decision_regions(self, X, y, resolution=0.02):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=np.array([cmap(idx)]),
                        marker=markers[idx], label=cl)

        plt.show()