# A 2D linear regression exercise

import random
import numpy as np
from matplotlib import pyplot as plt

# Dimension
D = 2

# For reproducibility
np.random.seed(10)

def sign(n):
    if n < 0: return -1
    elif n > 0: return 1
    else: return 0

def distance(x1, y1, x2, y2, x, y):
    # Calculate the distance of point (x, y) from the line defined by
    # the two points (x1, y1) and (x2, y2)
    # sign of the distance implies the location of the point relative to line
    # ie. under or above (-1/+1)
    d = (y - y2) * x1 - (x - x2) * y1 + x * y2 - y * x2
    return d


def fig():
    fig, ax = plt.subplots()
    return fig, ax

def draw(point, fig, ax, t=None):
    
    if t != None:
        plt.ion()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.plot([t[0], t[2]], [t[1], t[3]], c='green', label='Target function by two points')
        plt.pause(0.2)
        fig.canvas.draw()

    # Color points depending on location
    if point[-1] == -1 : c = 'red'
    else: c = 'blue'

    ax.scatter(point[0], point[1], c=c)

    plt.pause(0.2)
    fig.canvas.draw()



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

        data.append((x, y, sign(distance(t[0], t[1], t[2], t[3], x, y))))

        #if i == 0: 
        #    fig, ax = fig()
        #    draw(data[i], t, fig, ax)
        #elif i == n-1: 
        #    plt.clf()
        #    fig, ax = fig()
        #else: draw(data[i])

    # Format data
    inputs = [dt[:-1] for dt in data]
    outputs = [dt[-1] for dt in data]
    Y = np.array(outputs)
    X = np.array(inputs)
    return X, Y


class Metrics:
    """ Report metrics on the OLS"""

    def e_in(self):
        """ Calculate the in sample error of a hypothesis """
        X = np.c_[np.ones(self.data.shape[0]), self.data]
        #e_i = np.linalg.norm(np.dot(X, self.coef_) - self.target)**2/n
        Y_hat = [sign(label) for label in np.dot(X, self.coef_)]
        
        error = 0
        for y_hat, y in zip(Y_hat, Y):
            if y_hat != y: error += 1
        return error/n

    def print_stats():
        print(X_)
        print(self.coef_)
        print("In sample error: ", e_i/n)


class OLS(Metrics):
    """
    A linear regression algorithm
    """
    def __init__(self):
        self.coef_ = None
        self.data = None
        self.target = None


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
        data = np.c_[np.ones(X.shape[0]), X]
        return np.dot(data, self.coef_)


exp = 1000
n=100
H = []
# Run the experiment exp times while keeping track of the best hypothesis and 
print("Generating g's for problem 5")
print(2**6*'-')
# in sample error for each experiment
for experiment in range(exp):

    # Target function a set of four points
    t = gen_target()

    # input data and labels 
    X, Y = gen_data(n,t)

    # Find the coefficient 
    mlr = OLS()
    g = mlr.fit(X, Y)
    #mlr.predict(X)

    # Calculate the in sample error for the current hypothesis and store it along with
    # the hypothesis
    e_in = mlr.e_in()
    H.append((g, e_in))
    
    #print(f"In sample error for {experiment+1}. experiment {e_in}")

e = 0
for error in H:
    e += error[1]

print("In sample error: ", e/exp)
print('*'+2**6*'-'+'*')
print()


print("Generating 1000 fresh points for problem 6")
print(2**6*'-')
fresh_points = 1000

print('*'+2**6*'-'+'*')
