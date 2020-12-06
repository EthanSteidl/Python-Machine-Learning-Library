import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt



'''
Author: Ethan Steidl
The KNN class performs the K Nearest Neighbor learning algorithm
This algorithm can be found at https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
The KNN is a machine learning algorithm that classifies data with
the classification of the highest classification count of the k nearest 
neighbors in the training data set.
'''
class KNN(object):

    '''
    Author: Ethan Steidl
    Constructor of the KNN object. X is a set of training data for the model.
    y is the set of expected classifcations for the model.
    '''
    def __init__(self):
        self.X = np.empty
        self.y = np.empty


    '''
    Author: Ethan Steidl
    This function will take input data and find the absolute closest neighbor in
    the training data. This neighbor is returned and includes its classification,
    its features, and its index in the internal data set.

    param X - (DataFrame) Input data

    return Tuple(classification, featues of nearest point, index in training data)
    '''
    def _predict_single(self, X):
        """Return class label after unit step"""
        #distance = sqrt( (x1-x0)^2 + (y1-y0)^2 + ... )

        #for all the points from the user

        #for user_row in X:

        best_classification = 0
        best_distance = -1
        best_row = []
        best_index = 0
        count = 0
        #checked against all the data currently
        for stored_row, stored_class in zip(self.X, self.y):

            dist = np.linalg.norm(stored_row-X)

            if(dist < best_distance or best_distance < 0):
                best_distance = dist
                best_classification = stored_class
                best_row = stored_row
                best_index = count

            count += 1

        #answers.append(best_classification)

        #self.best_index = best_index

        return (best_classification, best_row, best_index)

    def fit(self, X, y):

        self.X = X
        self.y = y

        return

    '''
    Author: Ethan Steidl
    This function will calculate the k nearest neighbors for a single data point
    In addition, the index of the of each neighbor in the dataset is stored along
    with its values and classification. This is packaged into a list as a tuple
    and returned
    
    param X - (DataFrame) Input data
    param k - (Int) Count of nearby neigbors to use for classification

    return (ndArray) Array of 1 or -1 values. These values are the type of classification
    given by the model for that data instance
    '''
    def knn_group(self, X, k):

        #copy the variables from the class so they can be restored
        xcopy = self.X
        ycopy = self.y

        #list for storing nearby neighbors
        answers = []

        #Calculate the k nearest neighbors
        for i in range(0,k):
            answers.append(self._predict_single(X))

            #once a neighbor is found, remove it from the sample
            self.X = np.delete(self.X, answers[i][2], 0)
            self.y = np.delete(self.y, answers[i][2], 0)

        #restore all removed neighbors
        self.X = xcopy
        self.y = ycopy

        #return neighbors found
        return answers


    '''
    Author: Ethan Steidl
    This function will take input data and classify it acording to the KNN model.
    The result returned to the user will be a vector of -1 or 1 values. These values
    are classifications given to each instance during the prediction.

    param X - (DataFrame) Input data
    param k - (Int) Count of nearby neigbors to use for classification

    return (ndArray) Array of 1 or -1 values. These values are the type of classification
    given by the model for that data instance
    '''
    def predict(self, X, k=1):

        #list that stores the classification counts
        predictions = []

        for entry in X:
            # calculate the knn group for each data instance
            answers = self.knn_group(entry,k)
            sum = 0
            for item in answers:

                sum += item[0]

            #determine wich classification occured more often
            predictions.append(sum)

        #convert classification counts into -1 or 1 values and return the vector
        return np.where(np.asarray(predictions) >= 0, 1, -1)


    '''
    Author: Ethan Steidl
    This function will plot the input data X and color  differing data classifications different
    colors.  A data instance is passed into this function along with a k value. The function
    will run the KNN algorithm and show k nearest neighbors with a green asterisk. The data
    instance passed in will be classified and displayed on the plot as a sqare. The color of
    this square will shwo wich classification has been chosen for that data peice.

    param X - (DataFrame) Input data
    param y - (DataFrame) Training Set
    param variableX - (DataFrame) Single data instance
    param k - (Int) Count of nearby neigbors to use for classification

    return void
    '''
    def plot(self, X, y, variableX, k=1):


        #place each point of the graph where blue points have an expected result
        #of 1 and red has an expected result of -1 mark them with an x
        markers = ('x', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=np.array([cmap(idx)]),
                        marker=markers[idx], label=cl)


        #perform prediction and store the nearby points
        predictions = self.knn_group(variableX, k)

        #determine what classification the variableX is and set its color
        sum = 0
        for item in predictions:
            sum += item[0]
        if(sum >= 0):
            color = 'blue'
        else:
            color = 'red'

        #plot the variableX
        plt.plot(variableX[0], variableX[1], 's', color=color)

        #all nearby neighbors are marked with a green asterisk
        for item in predictions:

            color = 'green'
            plt.plot(item[1][0], item[1][1], '*', color='green')

        #graph
        plt.show()