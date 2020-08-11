# Learning from data homework exercise 7
# Percepton learning algorithm

# Target function generation
import numpy as np
import random
from matplotlib import pyplot as plt

D = 2  # Dimension of input space
N = 100  # Number of inputs
RUNS = 1000  # Iteration limit
DRAW = False


def draw_D(x):
    """
    Draws the generated input x
    """
    if not DRAW: return None, None, None
    fig, ax = plt.subplots()
    plt.title('Perceptron Learning Algorithm')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    plt.ion()  # Interactive mode for updating plot
    D_points = []
    for point in x:
        if point[-1] == 1:
            c = 'red'
        else:
            c = 'yellow'
        D_points.append(ax.scatter(point[1], point[2], c=c))
    fig.canvas.draw()
    plt.show(block=False)
    return fig, ax, D_points


def draw_f(x1, y1, x2, y2, ax):
    if not DRAW: return
    m = slope(x1, y1, x2, y2)
    x_ = np.arange(-5, 1)
    y = m * x_ + y1
    # ax.plot(y, c='orange', label='Target function via equation')
    ax.plot([x1, x2], [y1, y2], c='green', label='Target function by two points')
    ax.legend(loc='lower center')


def update_h(line, point, fig, ax):
    if not DRAW: return

    # Draw the hypothesis by line equation y = ax+b
    # x = point[1]-w[0]/w[1]*point[0]-w[2]/w[1]*point[2]
    # line.set_xdata(x)
    # y = point[2]-w[0]/w[2]*point[0]-w[1]/w[2]*point[1]
    # y = point[2]-w[0]/w[2]*x_-w[1]/w[2]*point[1]
    # print(f"New line coordinates: ({x}, {y})")
    # line.set_ydata(y)

    # Remove the previous line
    if line != None: pass

    # Draw the hypothesis by calculating two points
    x_ = np.linspace(-2, 2, 2)
    print("Weights: ", w)
    y1 = -w[0] / w[2] * x_[0] - w[1] / w[2] * point[1]
    y2 = -w[0] / w[2] * x_[1] - w[1] / w[2] * point[1]

    line = ax.plot(x_[0], x_[1], [y1, y2], c='orange')
    print(f"Line equation")
    fig.canvas.draw()
    # plt.pause(0.1)
    return line


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def gen_target_function():
    # Create random target function
    x1, y1 = np.random.uniform(-1, 1, 2)
    x2, y2 = np.random.uniform(-1, 1, 2)
    return x1, y1, x2, y2


def distance(x1, x2, y1, y2, x3, y3):
    # Calculate the distance of point (x3, y3) from the line defined by
    # the two points (x1, y1) and (x2, y2)
    # sign of the distance implies the location of the point relative to line
    # ie. under or above (-1/+1)
    d = (y3 - y2) * x1 - (x3 - x2) * y1 + x3 * y2 - y3 * x2
    return d


def sign(n):
    if n < 0:
        return -1
    elif n > 0:
        return 1
    else:
        return 0


def gen_input(x1, y1, x2, y2, n=None):
    """"
    Generate dataset for input x_n of the form
    1, x-coordinate, y-coordinate, index, sign
    """
    x = []
    if n:
        for i in range(n):
            x_, y_ = np.random.uniform(-1, 1, 2)
            s = sign(distance(x1, x2, y1, y2, x_, y_))
            x.append((1, x_, y_, i, s))
    else:
        for i in range(N):
            x_, y_ = np.random.uniform(-1, 1, 2)
            s = sign(distance(x1, x2, y1, y2, x_, y_))
            x.append((1, x_, y_, i, s))
    return x


def hypothesize(w, point):
    return sign(w[0] * point[0] + w[1] * point[1] + w[2] * point[2])


# Train the algorithm
iterations = []
miss = []
# Weights where w[0] is the threshold


def train(x, D_points, ax, fig):
    # random.shuffle(x)
    w = [0 for i in range(3)]
    learned = [False]

    def update_weights(w, x):
        i = 0

        while learned[0] == False:
            # If conv_count is not increased
            # the test passed without missclassifications
            # and learned is set to True
            conv_count = 0
            for point in x:

                # Test if h(x) = y, y being point[-1]
                #h = hypothesize(w, point)
                h = sign(w[0] * point[0] + w[1] * point[1] + w[2] * point[2])
                if h != point[-1]:
                    # Adjust the weights
                    w[0] = w[0] + point[-1] * point[0]
                    w[1] = w[1] + point[-1] * point[1]
                    w[2] = w[2] + point[-1] * point[2]
                    i += 1
                    conv_count += 1

                    # Plot h(x) calling update_h to draw the new
                    # line based on the updated weights
                    if i == 1:
                        line = None
                    if DRAW:
                        ax.plot(point[1], point[2], 'bo',
                                markersize='12.0', markerfacecolor='none')
                        fig.canvas.draw()
                    line = update_h(line, point, fig, ax)
                    # plt.pause(0.1)

                if point == x[-1] and conv_count == 0:
                    j = 0
                    #print("All points iterated, testing weights", w)
                    # TEST FOR CONVERGENCE
                    for pnt in x:
                        # print(f"current point {pnt}")
                        y = hypothesize(w, pnt)
                        # print("Weights classify as ", y)
                        if y != pnt[-1]:
                            j += 1
                            miss.append(j)
                            print("MISSCLASSIFICATION")
                    #print(f"Errors: {j} with rate {100 * j / N}%")
                    learned[0] = True

        # Learning done add iteration count to iterations
        iterations.append(i)

    while not learned[0]:
        update_weights(w, x)
    return w


def main():
    for i in range(RUNS):
        x1, y1, x2, y2 = gen_target_function()
        x = gen_input(x1, y1, x2, y2, n=None)
        fig, ax, D_points = draw_D(x)
        draw_f(x1, y1, x2, y2, ax)
        if DRAW: fig.canvas.draw()
        # plt.pause(0.1)
        weight = train(x, D_points, ax, fig)

        # Test how well the final weights classify a large sample
        # i.e. estimate P[f(x)=g(x)]
        if i == RUNS-1:
            count = 0
            n_ = 10000
            x = gen_input(x1, y1, x2, y2, n=n_)
            for i in range(n_):
                h = hypothesize(weight, x[i])
                if h != x[i][-1]: count += 1
            print("Last run: ", weight)

        if RUNS > 1: plt.close(fig=fig)
    total = 0
    for i in iterations:
        total += i
    print("Learned with average iterations of", total / len(iterations))
    print("Learning attempts: ", len(iterations))
    total = 0
    for i in miss: total += i
    print("P g neq f: ",count / n_)

    if DRAW:
        plt.ioff()
        plt.show()


main()

