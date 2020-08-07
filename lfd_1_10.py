"""
Exercise 1.10: Here is an experiment that illustrates the difference between a single
bin and a multiple bins. Run a computer simulation for flippin 1000 fair coins.
Flip each coin independently 10 times. Let's focus on 3 coins as follows:
c1 is the first coin flipped; c_rand is a coin you choose at random; c_min is the
coin that had the minimum frequency of heads (pick the earlier one in case of a tie).
Let v1, v_rand and v_min be the fraction of head syou obtain for the respective
three coins.
a) What is the mu for the three coins selected?
b) Repeat this entire experiment a large number of times (e.g. 100 000 runs) to get
several instances of v1, v_rand and v_min and plot the histograms of the
distributions v1, v_rand and v_min. Notice that which coins end up being c_rand and
c_min may differ from one run to another.
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from math import exp


class Coin:
    """ Represents a single coin"""

    def __init__(self):
        self.state = random.choice(('heads', 'tails'))

    def get_state(self):
        return self.state

    def __str__(self):
        return '{}'.format(self.get_state())
    
    def throw(self):
        self.state = random.choice(('heads', 'tails'))
        return self.state

    def __repr__(self):
        return self.__str__()


class Coinsimu:
    """ Heads/Tails -1/1 """
    # List of coins

    def __init__(self, n_coins, flips):
        """
         :param n_coins: number of cojins
         :param flips: number of flips per coin
         """
        self.n_coins = n_coins
        self.flips = flips

        # Add n_coins amount of coin objects and their ratio of heads to a list,
        # flip each coin flip times and count the ratio of heads for each coin
        self.coins = []
        for i in range(self.n_coins):
            h = 0
            coin = Coin()

            # Flip the coin  and count heads
            for j in range(self.flips):
                coin.throw()
                if coin.get_state() == 'heads': h += 1

            self.coins.append((coin, h/self.flips))

        # Pick the first coin, a random coin, and
        # the coin with the least heads

        self.c1 = self.coins[0]
        self.coins.sort(key=lambda coin: coin[1])
        self.c_min = self.coins[0]
        self.c_rand = self.coins[random.choice(range(0, self.n_coins))]
        
        

    def mu(self):
        return self.c1[1], self.c_rand[1], self.c_min[1]





flips = 10
n_coins = 10**3
runs = 10
v1 = 0
v_rand = 0
v_min = 0
for i in range(runs):
    simulation = Coinsimu(n_coins, flips)
    v1_, v_rand_, v_min_ = simulation.mu()
    v1 += v1_
    v_rand += v_rand_
    v_min += v_min_

print(v1/runs, v_rand/runs, v_min/runs)

fig, axs = plt.subplots(1, 2)

# We can set the number of bins with the `bins` kwarg
x = np.arange(3)
axs[0].bar(x, [v1/runs, v_rand/runs, v_min/runs])

e = np.linspace(0,1,100)
y = 2*np.exp(-2*e**2*1)
axs[1].plot(e, y, 'red')
y1 = 2*np.exp(-6*e**2)
axs[1].plot(e, y1, 'green')
axs[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
#axs.set_xticklabels(['First', 'Random', 'Least heads'])

plt.show()





