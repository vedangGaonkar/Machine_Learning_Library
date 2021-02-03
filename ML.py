"""
Author: Vedang D Gaonkar
Program: ML Library

First Update: Sep 25, 2020
First Update Contents: Perceptron, decision regions

Second Update: Oct 30, 2020
Second Update Contents: Linear Regression, Decision Stumps

Third Update: Dec 04, 2020
Third Update Contents: Prediction Intervals, Logistic Regression

Fourth Update: Dec 07, 2020
Fourth Update Contents: SVM (Support Vector Machine), KNN (K Nearest Neighbors)

"""
from random import shuffle, randint

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
from numpy.dual import norm

"""
Author: Vedang Gaonkar
Date Updated: Sep 25, 2020
Description: 
"""


class Perceptron(object):
    def __init__(self, rate=0.01, niter=10):
        self.errors = []
        self.rate = rate
        self.niter = niter
        self.weights = []

    # train data to create coefficients array
    def fit(self, x, y):
        self.weights = np.zeros(1 + x.shape[1])
        self.errors = []

        # Go through the errors list to see how it affects weights
        for i in range(self.niter):
            curr_errors = 0
            for xi, target in zip(x, y):
                # predict and use this in the 1 or -1 iter value
                iter_value = self.rate * (target - self.predict(xi))
                self.weights[1:] += iter_value * xi
                self.weights[0] += iter_value
                curr_errors += int(iter_value != 0.0)
            self.errors.append(curr_errors)
            # print(" Errors " + curr_errors.__str__())

            # Breaks out of the loop when errors converge
            if curr_errors == 0:
                break
        return self

    def net_input(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


"""
Authors: Christer Karlsson, Vedang Gaonkar
Date Updated: Sep 25, 2020
"""


def plot_decision_regions(X, y, classifier, resolution=0.01):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=np.atleast_2d(cmap(idx)),
                    marker=markers[idx], label=cl)


'''
Author: Vedang D Gaonkar
Date Updated: Oct 30, 2020
Resource Credits:
Anirudh Sinha (towardsdatascience.com)
'''


class LinearRegression:

    @staticmethod
    def fit(X, Y):
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)

        # Here, the shapes are just the dimenstions
        # For this algorithm, we only have one dimensional data

        xShape = X.shape

        # the coefficient matrix will only have a 0th element for d = 1
        numVariable = xShape[1]
        wMatrix = np.random.normal(0, 1, (numVariable, 1))
        intercept = np.random.rand(1)

        # 500 can be changed to any epoch value you want
        for i in range(500):
            # summing the partial derivatives of costs w.r.t. weight (coeff) and intercept respectively.
            # Find the partial derivatives explanation in the resource mentioned above
            a = np.sum(np.multiply(((np.matmul(X, wMatrix) + intercept) - Y), X)) * 2 / xShape[0]
            b = np.sum(((np.matmul(X, wMatrix) + intercept) - Y)) * 2 / xShape[0]
            wMatrix -= 0.1 * a
            intercept -= 0.1 * b

        return wMatrix, intercept

    @staticmethod
    def predict(X, weight_matrix, intercept):
        product = np.matmul(np.array(X).reshape(-1, 1), weight_matrix) + intercept
        return product


'''
Author: Vedang D Gaonkar
Date Updated: Dec 07, 2020
'''


class LogisticRegression:

    # Initialize the learning rate and go from there
    def __init__(self):
        self.rate = 0.3
        self.intercept = 0
        self.w1 = 0
        self.w2 = 0

    # fit function to train the explanatory data
    def fit(self, X, y):
        for _ in range(0, 150):
            # Go through each row in the data
            for i in range(len(X)):
                x1 = X[i][0]
                x2 = X[i][1]

                # predict for the current pair of data points (2D)
                prediction = self.find_sigmoid(x1, x2)
                error = y[i] - prediction

                # update intercept and weights wrt error, prediction, rate
                self.w1 = self.w1 + self.rate * error * prediction * (1.0 - prediction) * x1
                self.w2 = self.w2 + self.rate * error * prediction * (1.0 - prediction) * x2
                self.intercept = self.intercept + self.rate * error * prediction * (1.0 - prediction)

    # predict for independent variable
    def predict(self, x1, x2):
        return self.find_sigmoid(x1, x2)

    # Math function to find sigmoid
    def find_sigmoid(self, x1, x2):
        return 1.0 / (1.0 + np.exp(- (self.intercept + self.w1 * x1 + self.w2 * x2)))


'''
Author: Vedang D Gaonkar
Date Modified: Dec 07, 2020
'''


def PredictionIntervals(x, y):
    # This acts as a wrapper around linear regression
    linearRegression = LinearRegression()

    # train the data
    wMatrix, intercept = linearRegression.fit(x, y)

    # This is basically just the prediction part simplified
    linearPrediction = intercept + wMatrix[0] * x

    # Choose a x, y, and interval production output
    i = randint(0, 5)
    x_in = x[i]
    y_out = y[i]
    prediction_out = linearPrediction[i]

    # Find the standard deviation
    sum_errs = sum((y - linearPrediction) ** 2)
    sd = np.sqrt(1 / (len(y) - 2) * sum_errs)

    # calculate and print prediction interval
    # 1,96 is the standard dev value for 95% accuracy
    interval = 1.96 * sd
    print('The prediction interval is: %.3f' % interval)

    # Define the range of the interval
    lower, upper = prediction_out - interval, prediction_out + interval

    # 95% because of the 1.96 value
    print('Predicting 95%% chance of the value being between %.3f and %.3f' % (lower, upper))
    print('Expected value: %.3f' % y_out)

    return linearPrediction, prediction_out, x_in, interval


# Function to find the euclidean distance between two points
def euclideanDistance(a, b):
    return norm(a - b)


'''
Author: Vedang D Gaonkar
Date Updated: Dec 06, 2020
'''


class KNN:

    # initialize k and distance values
    # Currently k = 1 for the 1-NN problem
    def __init__(self, k=1, distance_metric=euclideanDistance):
        self.k = k
        self.distance = distance_metric
        self.data = None

    # Train the data with dependant and independent vectors
    def train(self, X, y):
        # Checker to convert into lists
        if type(X) == np.ndarray:
            X, y = X.tolist(), y.tolist()

        # set data attribute containing instances and labels
        self.data = [X[i] + [y[i]] for i in range(len(X))]

    # Predict the suitable class based on k-nearest neighbors
    def predict(self, a):
        # List of neighbors
        neighbors = []

        # Distances of the classes from the current
        distances = {self.distance(x[:-1], a): x for x in self.data}

        # Find classes of instances with k shortest distances
        # You can sub this with your own max heap implementation
        for key in sorted(distances.keys())[:self.k]:
            neighbors.append(distances[key][-1])

        # Max is the most common vote
        return max(set(neighbors), key=neighbors.count)


'''
Author: Vedang D Gaonkar
Date Updated: Oct 30, 2020
Resource Credits:
Google Developers YouTube Channel, 'Machine Learning in action' by Peter Harrington
'''


class DecisionStumps:

    @staticmethod
    def stumpClassify(dataMatrix, dimension, threshold, thresholdInequality):
        # Dimension check
        retArray = np.ones((np.shape(dataMatrix)[0], 1))
        # Implement thresholds to classify into -1 or 1
        # Number of dimensions also dictates this
        if thresholdInequality == 'lt':
            retArray[dataMatrix[:, dimension] <= threshold] = -1.0
        else:
            retArray[dataMatrix[:, dimension] > threshold] = -1.0
        return retArray

    # Function to build the best stump
    def buildStump(self, dataArray, classLabels, D, runNum):

        # Utility to reduce clutter in the terminal
        global f
        if runNum == 1:
            f = open("decisionStumpsResults1.txt", "wt")
            f.write("Petal Width vs Petal Length\n\n")

        if runNum == 2:
            f = open("decisionStumpsResults2.txt", "wt")
            f.write("Sepal Width vs Sepal Length\n\n")

        # Typecast
        dataMatrix = np.mat(dataArray)
        labelMat = np.mat(classLabels).T

        # Determine the number of iterations/range to run
        m, n = np.shape(dataMatrix)
        numSteps = 10.0

        # Dictionary to store stumps
        stumpMap = {}
        # Best class estimate for the data
        bestClassEstimate = np.mat(np.zeros((m, 1)))
        minError = np.inf

        # Iterate through the range to provide
        for i in range(n):
            rangeMin = dataMatrix[:, i].min()
            rangeMax = dataMatrix[:, i].max()

            stepSize = (rangeMax - rangeMin) / numSteps

            # Inner loop to update errors
            for j in range(-1, int(numSteps) + 1):
                for unequal in ['lt', 'gt']:
                    threshVal = (rangeMin + float(j) * stepSize)
                    predictedValues = self.stumpClassify(dataMatrix, i, threshVal, unequal)

                    errorArray = np.mat(np.ones((m, 1)))
                    errorArray[predictedValues == labelMat] = 0

                    # Weighted Error calculation
                    weightedError = D.T * errorArray

                    f.write("Split:\nDimension: %d, Threshold: %.2f, "
                            "Threshold Ineqality: %s, The weighted error is %.3f\n\n" % (
                                i, threshVal, unequal, weightedError))

                    # new minimum attributes
                    if weightedError < minError:
                        minError = weightedError
                        bestClassEstimate = predictedValues.copy()
                        stumpMap['dim'] = i
                        stumpMap['thresh'] = threshVal
                        stumpMap['ineq'] = unequal

        return stumpMap, minError, bestClassEstimate


'''
# >> MODEL TRAINING << #
def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = regularization_strength * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost


# I haven't tested it but this same function should work for
# vanilla and mini-batch gradient descent as well
def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])  # gives multidimensional array

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
        dw += di

    dw = dw / len(Y_batch)  # average
    return dw


def sgd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)

        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights
'''

'''
Author: Vedang D Gaonkar
Date Updated: Dec 07, 2020
Resource Credits: Chandler Severson Github
'''


def svm(x, y):
    learning_rate = 1

    # Initialize the SVMs weight vector with all zeros
    w = np.zeros(len(x[0]))

    epochs = 100000

    missclassifications = []

    # Train and Gradient Descent
    for epoch in range(1, epochs):
        error = 0
        calculate_cost_gradient(w, x, y, 1000)
        for i, n in enumerate(x):
            if (y[i] * np.dot(x[i], w)) < 1:
                w = w + learning_rate * ((x[i] * y[i]) + (-2 * (1 / epoch) * w))
                error = 1
            else:
                # correct classified weight update
                w = w + learning_rate * (-2 * (1 / epoch) * w)
        missclassifications.append(error)

    return w, missclassifications


def calculate_cost_gradient(W, X, Y, regularize=None):
    dis = 1 - (Y * np.dot(X, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(dis):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (regularize * Y[ind] * X[ind])
        dw += di

    dw = dw / len(Y)  # average
    return dw
