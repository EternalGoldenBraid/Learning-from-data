# A 2D linear regression exercise

import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from copy import deepcopy
import os
from time import time


# Dimension
D = 2
PAUSE = 0.001

# For reproducibility
#np.random.seed(3)

def sign(n):
    if n < 0.01: return -1
    elif n > -0.01: return 1
    else: return 0

def slope_intercept(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y2 - slope*x2
    return slope, intercept

def distance(x1, y1, x2, y2, x, y):
    # Calculate the distance of point (x, y) from the line defined by
    # the two points (x1, y1) and (x2, y2)
    # sign of the distance implies the location of the point relative to line
    # ie. under or above (-1/+1)
    #d1 = (y - y2) * x1 - (x - x2) * y1 + x * y2 - y * x2
    d = (x-x2)*(y2-y1) - (y-y2)*(x2-x1)
    target = [x2-x1, y2-y1]
    point = [x-x1, y-y1]
    d = np.cross(point, target)
    return d


def fig():
    fig, ax = plt.subplots(1,1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return fig, ax

def draw(fig, ax, points=None, t=None, h=None, linear_regression=False):
    """
    Draw a figure illustrating the Percpetron learning algorithm and linear regression.
    param point: Not None if adding scatter of training data.
    param t: Not None if adding target function.
    param h: Not none if adding a hypothesis.
    param linear_regression: True if the initial weights are found by linera regression.
    """

    h_line = None
    x_vals = np.array(ax.get_xlim())
    x_vals = np.array((-1.0, 1.0))
    
    if t != None:
        # Add the target function
        plt.ion()
        slope, intercept = slope_intercept(t[0], t[1], t[2], t[3])
        y_vals = intercept + slope*x_vals
        label='Target function by two points'
        ax.plot(x_vals, y_vals, '-', c='green', label=label)

        ax.scatter(t[0], t[1], c='black')
        ax.scatter(t[2], t[3], c='black')


    elif points != None:
        # Add scatter plot

        # Color points depending on location
        for point in points:
            if point[-1] == 1 : 
                c = 'red'
                label = 'above'
            else: 
                c = 'blue'
                label = 'under'

            label = None
            ax.scatter(point[0][0], point[0][1],
                    label=label, c=c)

    else:
        # Add the hypothesis
        if linear_regression == True: 
            label='Hypothesis given by OLS'
            s = ':'
            c = 'r'
        else:
            label='Hypothesis given by PLA'
            c = 'y'
            s= '--'

        y_vals = h[0] + h[1]*x_vals
        h_line, = ax.plot(x_vals, y_vals, s, label=label)
        plt.pause(PAUSE)

    plt.pause(PAUSE)
    fig.canvas.draw()
    ax.legend(loc='upper left')

    if h_line != None: return h_line


def gen_target():
    """ Create and return a random target function
    """
    x1, y1 = np.random.uniform(-1, 1, 2)
    x2, y2 = np.random.uniform(-1, 1, 2)
    return x1, y1, x2, y2

def gen_data(n, t):
    """ Generate data points
    return the input matrix and output vector
    param n: Number of data points to generate
    param t: tuple target function with two coordinates x1, y1, x2, y2
    """
    data = []
    for i in range(n):
        x, y = np.random.uniform(-1,1, 2)
        s = sign(distance(t[0], t[1], t[2], t[3], x, y))
        data.append((x, y, s))

    # Format data
    inputs = [dt[:-1] for dt in data]
    outputs = [dt[-1] for dt in data]
    Y = np.array(outputs)
    X = np.array(inputs)
    return X, Y


class Metrics:
    """ Report metrics on the OLS"""

    def e_in(self, X, Y, n):
        """ Calculate the in sample error of a hypothesis """
        X = np.c_[np.ones(self.data.shape[0]), self.data]
        Y_hat = [sign(label) for label in np.dot(X, self.coef_)]
        
        error = 0
        for y_hat, y in zip(Y_hat, Y):
            if y_hat != y: error += 1
        return error/n

    def e_out(self, X_test, Y_test, n_test):
        """ Calculate the out of sample error of a hypothesis """
        
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        Y_hat = [sign(prediction) for prediction in np.dot(X_test, self.coef_)]
        error = 0
        for label, prediction in zip(Y_test, Y_hat):

            if label != prediction: error += 1

        return error/n_test

    def e_avg(self, exp, n, n_test):
        """ Print out the average in sample error 
        and out of sample error over exp number of experiments.
        param exp: Number of experiments to run
        param n: Number of sample points to generate
        param n_test: Number of test points to test out of sample error on
        param G_OLS: The set of all the final hypotheses 
        """
        print("Generating final hypothesise for problem 5")
        print(2**6*'-')


        # in sample error for each experiment
        for experiment in range(exp):
        
            # Target function a set of four points
            t = gen_target()
        
            # Training data and their labels
            X, Y = gen_data(n,t)
        
            # Find the coefficients
            #mlr = OLS()
            g = self.fit(X, Y)
        
            # Calculate the in sample error for the current hypothesis 
            # and store it along with the hypothesis
            e_in_ = self.e_in(X, Y, n)

            # Generate testing points and evaluate with respect to the target function
            X_test, Y_test = gen_data(n_test, t)

            e_out_ = self.e_out(X_test, Y_test, n_test)

            G_OLS.append((g, e_in_, e_out_))

            
        in_sample_error = 0
        out_sample_error = 0
        for error in G_OLS:
            in_sample_error += error[1]
            out_sample_error += error[2]
        
        print(f"Average in sample error for {exp} experiments: " \
                f"{in_sample_error/exp:.4f}")
        print(f"Average out of sample error for {exp} experiments: " \
                f"{out_sample_error/exp:.4f}")
        print('*'+2**6*'-'+'*')
        print()


    def perc_error(self, exp, n, n_test, linear_regression = False):
        """ 
        Print out the average number of iterations required 
        for convergence of the PLA.
        param exp: Number of experiments to run
        param n: Number of sample points to generate
        param n_test: Number of test points to test out of sample error on
        param G_PLA: The set of all the final hypotheses 
        param linear_regression: Dictates which set of weights to use
        """
        print("Generating final hypothesise for problem 7")
        print(2**6*'-')
    
        iter_converge = []
    
        self.fig_, self.ax = fig()

        # in sample error for each experiment
        for experiment in range(exp):
        
            # Target function a set of four points
            t = gen_target()
            draw(self.fig_, self.ax, t=t)

            # Training data and their labels
            X, Y = gen_data(n,t)
            draw(self.fig_, self.ax, points=zip(X,Y))
        
            # Find the coefficients with linear regression
            self.fit(X, Y)
            draw(self.fig_, self.ax, h=self.weights_OLS, linear_regression=True)


            
            # Use the perceptron to perfectly classify the data
            # and store the number of iterations required for convergence
            iter_converge.append(self.train(X, Y, linear_regression=True))
            draw(self.fig_, self.ax, h=self.weights)

            plt.cla()
        
        plt.ioff()
        plt.show()
        count = 0
        for i in iter_converge:
            count += i
    
        print(f"Average number of iterations {count/exp}")


class OLS(Metrics):
    """
    A linear regression algorithm
    """
    def __init__(self):
        self.coef_ = None
        self.data = None
        self.target = None
        self.t = None


    def fit(self, X, Y):
        """
        Generate the best fit hypothesis
        param X: A 2D numpy array for inputs
        param Y: A 1D numpy array for label
        """

        # Training data and labels
        self.data = X
        self.target = Y

        # Add bias to data
        X = np.c_[np.ones(X.shape[0]), X] # Add ones

        # Analytical solution
        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X.T, Y)
        self.coef_ = np.dot(inverse_xTx, xTy)
        return self.coef_


    def predict(self, X):
        """ Output the prediction of a final hypothesis given an input X
        param X: 2D Numpy array
        """
        #data = np.c_[np.ones(X.shape[0]), X]
        return np.dot(data, self.coef_[1:])


class PercLinear(Metrics):
    """ 
    A perceptron learning algorithm that uses initial weights found by linear regression
    as a starting point
    """

    def __init__(self):
        """ 
        Set initial weights to zeroes 
        for D + 1 see the definition of the Perceptron Learning Algorithm 
        """
        self.weights = [0 for i in range(D + 1)]
        self.weights_OLS = []
        self.learned = False

    def fit(self, X, Y):
        """
        Generate the best fit hypothesis
        param X: A 2D numpy array for inputs
        param Y: A 1D numpy array for label
        """

        # Training data and labels
        self.data = X
        self.target = Y

        # Add bias to data
        X = np.c_[np.ones(X.shape[0]), X] # Add ones

        # Analytical solution
        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X.T, Y)
        self.weights_OLS = np.dot(inverse_xTx, xTy)

    def train(self, X, Y, linear_regression = False):
        """ 
        Traing and find the weights that linearly separate the data 
        with and without the help of linear regression.
        param X: Sample data.
        param ax: PyPlot axis to draw on.
        """

        time_a = time()
        
        if linear_regression == False: 
            w = self.weights
        else: w = self.weights_OLS

        # Total number of iterations to adjust weights accordingly.
        i = 0

        # Add bias term to data
        X = np.c_[np.ones(self.data.shape[0]), self.data]

        self.learned = False
        while self.learned == False:

            # Draw the current hypothesis
            h_line = draw(self.fig_, self.ax, h=w, linear_regression=False)

            # Counts iterations required for convergence
            # If training passes without an increase in the count we are in business.
            conv_count = 0

            seta = np.c_[X,Y]
            np.random.shuffle(seta)
            X, Y = (seta[:, :-1], seta[:, -1])
            for point, label in zip(X, Y): 

                # Test if the hypothesis missclassifies the point
                h = sign(w[0] * point[0] + w[1] * point[1] + w[2] * point[2])

                # Highlight the point being evaluated againts
                focus, = self.ax.plot(point[1], point[2], 'bo',
                        markersize='12.0', markerfacecolor='none')
                plt.pause(PAUSE)
                self.fig_.canvas.draw()

                if h != label:
                    # Adjust the weights
                    #if i > 900000:
                    #    print("Adjusting weights from: ", w)
                    #    print("h: ", w[0] * point[0] + w[1] * point[1] + w[2] * point[2])
                    w[0] = w[0] + label * point[0]
                    w[1] = w[1] + label * point[1]
                    w[2] = w[2] + label * point[2]
                    #if i > 900000: 
                    #    print("To: ", w)
                    #    print("h: ", w[0] * point[0] + w[1] * point[1] + w[2] * point[2])
                    #    input()
                    i += 1
                    conv_count += 1

                # If current point is the last point and no missclassifications
                # have occured, set learned to True. We have learned.
                if h == label and conv_count == 0:
                    idx = 0
                    for point, label in zip(X, Y):
                        h_test = sign(w[0] * point[0] + w[1] * point[1] + w[2] * point[2])
                        if label != h_test:
                            idx =+ 1
                    if idx == 0: 
                        self.learned = True
                        print(f"'Learned' {len(Y)} points")
                        plt.pause(4)
                        return i

                # Remove the highlight of current test point
                focus.remove()

            # Remove the current hypothesis
            if h_line != None: h_line.remove()

        




mlr = OLS()
mlr_pla = PercLinear()

G_OLS= [] # An empty set for storing the final hypothesise
# Calculate the average in sample error for 1000 experiments on N training points
# and out of sample error on n_test fresh data points.
experiments = 1000
N = 10
n_test = 10
mlr.e_avg(experiments, N, n_test)
N = 10
mlr_pla.perc_error(experiments, N, n_test)




