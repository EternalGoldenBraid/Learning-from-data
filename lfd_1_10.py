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
import matplotlib as plt
import numpy as np


class Coin:
    """ Represents a single coin"""

    def __init__(self):
        self.state = random.choice(('heads', 'tails'))

    def get_state(self):
        return self.state

    def __str__(self):
        return '{}'.format(self.get_state())
    
    def throw(self):
        return random.choice(('heads','tails'))

    def __repr__(self):
        return self.__str__()


class Coinsimu:
    """ Heads/Tails -1/1 """
    # List of coins

    def __init__(self, n_coins, flips, runs):
        """
         :param n_coins: number of cojins
         :param flips: number of flips per coin
         :param runs: iterations or runs of the experiment
         """
        self.n_coins = n_coins
        self.flips = flips
        self.runs = runs

        # Add n_coins amount of coin objects and their ratio of heads to a list,
        # flip each coin flip times and count the ratio of heads for each coin
        self.coins = []
        for i in range(self.n_coins):
            h = 0
            coin = Coin()
            self.coins.append((coin, h/self.flips)
            for j in range(self.flips):
                self.coins[i].throw()
                if self.coins[i].get_state() == 'heads': h += 1


        # Pick the first coin, a random coin, and
        # the coin with the least heads

        self.c1 = self.coins[0]
        self.c_rand = self.coins[random.choice(0, len(self.coins))




simulation = Coinsimu(10**3, 10, 10**5)

