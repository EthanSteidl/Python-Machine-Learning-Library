import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

'''
Author: Ethan Steidl
The Axis Aligned Rectangle class performs the Axis Aligned Rectangle learning
algorithm. This learning algorithm creates a d-dimensional rectangle around the
training data. The rectangle created has the highest number of correct classifications 
inside the rectangle in addition to classifying wrong values outside the rectangle
The points of the rectangle consist of two points from the data set.
'''
class AxisAlignedRectangle():
    '''
    Constructor for class
    corners - (np.ndarray) numpy array holding the two corners of a rectangle
    accuracy - (float) percentage how how correct the model is for the training data
    dimensions - (int) how many dimensions the model was trained on
    '''

    def __init__(self):

        self.corners = np.empty
        self.accuracy = 0.0
        self.dimensions = 0

    '''
    Will alter corners to produce a model that maps the input data X
    to the training data y. After the model is fit, accuracy is calculated
    based off how many points are correcly classified

    param X - (DataFrame) Input data
    param y - (DataFrame) Desired output

    return self - returns this object as a model since corners and accuracy are stored
    '''

    def fit(self, X, y):
        # calculates dimensions of problem
        self.dimensions = X.shape[1]

        accuracy = 0  # best accuracy found
        best = [0, 0]  # pair of points corresponding to accuracy


        # for each combination of points
        for i in range(0, X.shape[0]):
            for j in range(i + 1, X.shape[0]):

                # calcluate the accuracy of the rectangle made by the points
                temp_accuracy = self.solve_accuracy(X[i], X[j], X, y)

                # if the accuracy is better than the current
                # update accuracy and best points
                if (temp_accuracy > accuracy):
                    accuracy = temp_accuracy
                    best = np.array([X[i], X[j]])

        # set class variables for best points and accuracy
        self.corners = best
        self.accuracy = accuracy

        return self

    '''
    Will retrun an accuracy score of a given D dimensional rectangle
    defined by p1 and p2 (points of D dimensions) for the testing data X 
    with expected results y

    param p1 - (ndArray) Array of size D representing a D dimensional point
    param p2 - (ndArray) Array of size D representing a D dimensional point
    param X - (DataFrame) Input data
    param y - (DataFrame) Desired output

    return (float) accuracy score of given rectangle
    '''
    def solve_accuracy(self, p1, p2, X, y):

        # order points in acending order by their first dimension, then second, ... till D
        points = np.array([p1, p2])
        points.sort(axis=0)

        # score of rectangle
        score = 0

        # for each point, check if it is in the rectangle bounded by p1 and p2
        # if it is and the expected result is 1, add 1 to score
        # if it is not and the expected result is -1, add 1 to score
        for point, answer in zip(X, y):
            inside = True
            for pos in range(0, X.shape[1]):

                if (points[0][pos] > point[pos] or point[pos] > points[1][pos]):
                    inside = False
            if (inside == True and answer == 1):
                score += 1
            elif (inside == False and answer == -1):
                score += 1

        # returns the score / amount of points.  This will always be between 0 and 1.0 inclusive
        return score / X.shape[0]


    '''
    Calcualtes whether each insance in X should be classified as pass(1) or fail(-1)
    If the instance is inside the rectangle classifier it is deemed a pass(1).
    If the instance is outside the rectangle classifier it is deemed a fail(-1).
    
    param X - (DataFrame) Input data
    
    return (np.ndArray) A one dimensional array of 1 or -1 values the length of X
    '''
    def predict(self, X):

        #sort the corners such that the lowest index has the lowest first dimension
        corners = self.corners
        corners.sort(axis=0)

        #holds result of classifier
        values = []

        #for each instance in X calculate if the instance is inside the rectangle
        #if it is, assign it the value of 1
        #if it is not, assign it the value of -1
        for point in X:
            inside = True
            for pos in range(0, X.shape[1]):
                if (corners[0][pos] > point[pos] or point[pos] > corners[1][pos]):
                    inside = False
            if (inside == True):
                values.append(1)
            elif (inside == False):
                values.append(-1)

        #return the result as a numpy array
        return np.array(values)

    '''
    Plots each instance from X where the color of the point represents the expected
    result from y.  The rectangle classifier will be plotted ontop of the graph
    in blue.
    
    param X - (DataFrame) Input data
    param y - (DataFrame) Desired output
    
    return none
    '''
    def plot(self, X, y):

        #place each point of the graph where blue points have an expected result
        #of 1 and red has an expected result of -1
        for item, answer in zip(X, y):
            c = 'red'
            if (answer == 1):
                c = 'blue'

            plt.plot(item[0], item[1], 'x', color=c)

        #creates an overlay for the rectangle
        ax = plt.gca()

        #solve for variables needed when graphing the rectangle
        width = (self.corners[1][0] - self.corners[0][0])
        height = (self.corners[1][1] - self.corners[0][1])
        cornerx = (self.corners[0][0])
        cornery = (self.corners[0][1])

        #place rectangle on graph
        rect = matplotlib.patches.Rectangle((cornerx, cornery), width, height, linewidth=1, edgecolor='r')
        ax.add_patch(rect)

        #graph
        plt.show()
