# A 2D nonlinear transformation and linear regression exercise

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

def target(x1, x2):
    """ The unknown target function """

    return sign(x1**2+x2**2-0.6)

def target_tran(x1, x2, w):
    """ Function for testing the hypothesis obtained from the nonlinear feature vector """
    
    return sign(w[0]*x1+w[2]*x2+w[3]*x1*x2+w[4]*x1**2+w[5]*x2**2)
def gen_data(n, transform=False):
    """ Generate data points
    return the input matrix and output vector
    param n: Number of data points to generate
    """
    data = []
    for i in range(n):
        x, y = np.random.uniform(-1,1, 2)
        s = target(x, y)
        if transform: data.append((x, y, x*y, x**2, y**2, s))
        else: data.append((x,y,s))

    # Generate simulated noice by flipping the sign of 10% of the data
    noice_idx = np.random.choice(n, size=int(n/10))

    for idx in noice_idx:
        point_sign = data[idx][-1]
        if point_sign == -1: point_sign = 1
        else: point_sign = -1

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

    def e_transform(self, g, X, Y):
        """ Trasformed feature vector: 
            Compare g to options and calculate out of sample error """

        #n_options = 5
        def a(x1, x2):
            return sign(-1-0.05*x1+ 0.08*x2+ 0.13*x1*x2+ 1.5*x1**2+ 1.5*x2**2)
        def b(x1, x2):
            return sign(-1-0.05*x1+ 0.08*x2+ 0.13*x1*x2+ 1.5*x1*2+ 15*x2**2)
        def c(x1, x2):
            return sign(-1-0.05*x1+ 0.08*x2+ 0.13*x1*x2+ 15*x1**2+ 1.5*x2**2)
        def d(x1, x2):
            return sign(-1-1.5*x1+ 0.08*x2+ 0.13*x1*x2+ 0.05*x1**2+ 0.05*x2*2)
        def e(x1, x2):
            return sign(-1-0.05*x1+ 0.08*x2+ 1.5*x1*x2+ 0.15*x1**2+ 0.15*x2**2)

        def g_test(X, Y):
            out_error = 0 # Used for calculating out of sample error
            X = np.c_[np.ones(X.shape[0]), X] # Add ones
            for point, label in zip(X, Y):
                diff_a = int(a(point[1], point[2]) != label)
                diff_b = int(b(point[1], point[2]) != label)
                diff_c = int(c(point[1], point[2]) != label)
                diff_d = int(d(point[1], point[2]) != label)
                diff_e = int(e(point[1], point[2]) != label)

                E_nonlin['a'] += diff_a/n_test
                E_nonlin['b'] += diff_b/n_test
                E_nonlin['c'] += diff_c/n_test
                E_nonlin['d'] += diff_d/n_test
                E_nonlin['e'] += diff_e/n_test

                # Measure the performance of estimated g i.e. out of sample error
                #E_nonlin['g'] += int(label != target_tran(point[1], point[2], g))
                out_error += int(label != target_tran(point[1], point[2], g))

            E_nonlin['g'] = out_error/n_test


        g_test(X, Y)

    def e_avg(self, exp, n, n_test, transform=False):
        """ Print out the average in sample error 
        and out of sample error over exp number of experiments.
        param exp: Number of experiments to run
        param n: Number of sample points to generate
        param n_test: Number of test points to test out of sample error on
        param G_OLS: The set of all the final hypotheses 
        """
        
        # in sample error for each experiment
        for experiment in range(exp):
        

            # Training data and their labels
            X, Y = gen_data(n, transform=transform)
        
            # Find the coefficients
            #mlr = OLS()
            g = self.fit(X, Y)

            # If we're transforming proceed for an alternative error measure and return
            if transform:
                X_trans, Y_trans = gen_data(n, transform=transform)
                self.e_transform(g, X_trans, Y_trans)

                # On last iteration print results
                if experiment == exp-1:
                    # Compare to the hypothesise for the transformed nonlinear
                    print("""
                    The hypothesise to consider are:
a) g(x1,x2) = sign(−1−0.05x1+ 0.08x2+ 0.13x1x2+ 1.5x21+ 1.5x22)
b) g(x1,x2) = sign(−1−0.05x1+ 0.08x2+ 0.13x1x2+ 1.5x21+ 15x22)
c) g(x1,x2) = sign(−1−0.05x1+ 0.08x2+ 0.13x1x2+ 15x21+ 1.5x22)
d) g(x1,x2) = sign(−1−1.5x1+ 0.08x2+ 0.13x1x2+ 0.05x21+ 0.05x22)
e) g(x1,x2) = sign(−1−0.05x1+ 0.08x2+ 1.5x1x2+ 0.15x21+ 0.15x22)
""")
                    print(f"""
The hypothesis closest to what we learned is:
a: {E_nonlin['a']/exp:.3f},
b: {E_nonlin['b']/exp:.3f},
c: {E_nonlin['c']/exp:.3f},
d: {E_nonlin['d']/exp:.3f},
e: {E_nonlin['e']/exp:.3f}""")
                    print(min(E_nonlin, key=E_nonlin.get))
                    print(f"""
    Out of sample error: {E_nonlin['g']/exp:.3f}
                         b: {abs(0.1-E_nonlin['g']/exp):.3f}
                         c: {abs(0.3-E_nonlin['g']/exp):.3f}
                         """)
                

            else:
                # Calculate the in sample error for the current hypothesis 
                # and store it along with the hypothesis
                e_in_ = self.e_in(X, Y, n)

                # Generate testing points and evaluate with respect to the target function
                X_test, Y_test = gen_data(n_test)

                e_out_ = self.e_out(X_test, Y_test, n_test)

                G_OLS.append((g, e_in_, e_out_))

            
                in_sample_error = 0
                out_sample_error = 0
                for error in G_OLS:
                    in_sample_error += error[1]
                    out_sample_error += error[2]
                
                if experiment == exp-1:
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
        """
        Various parameters to be adjusted by further function calls from Metrics or OLS
        param coef: The weights of the hypothesis.
        param data: Generated labeled data.
        param target: The generated target function.
        param transform = A boolead dictating wheter to transform the training data.
        """
        self.coef_ = None
        self.data = None
        self.target = None
        self.t = None
        self.transform = None


    def fit(self, X, Y):
        """
        Generate the best fit hypothesis
        param X: A 2D numpy array for inputs
        param Y: A 1D numpy array for label
        return: The approximate weights.
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


mlr = OLS()
G_OLS= [] # An empty list for storing the final hypothesise
# An empty dict for storing the differences to each option
E_nonlin = {
        'a' : 0,
        'b' : 0,
        'c' : 0,
        'd' : 0,
        'e' : 0,
        'g' : 0,
        } 
# Calculate the average in sample error for 1000 experiments on N training points
# and out of sample error on n_test fresh data points.
experiments = 1000
N = 100
n_test = 1000
mlr.e_avg(experiments, N, n_test)

experiments = 1000
N = 1000
n_test = 1000
mlr.e_avg(experiments, N, n_test, transform=True)
