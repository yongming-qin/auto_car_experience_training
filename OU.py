import random
import numpy as np 

class OU(object):

    def function(self, x, theta, mu, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)