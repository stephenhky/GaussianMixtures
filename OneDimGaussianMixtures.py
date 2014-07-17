__author__ = 'hok1'

import numpy as np
from scipy.stats import norm
from functools import partial

sample_parameters = [{'weight': 0.4, 'mean': 0, 'stdev': 1},
    {'weight': 0.6, 'mean': 2, 'stdev': 1.2}]

def density(x, parameters):
    zs = map(lambda p: (x-p['mean'])/p['stdev'], parameters)
    return sum(map(lambda z, param: norm.pdf(z)*param['weight'], zs, parameters))

def densities(xs, parameters):
    return np.array(map(partial(density, parameters=parameters), xs))

def MetropolisMarkovChain(distfunc, nSamples, startX=0):
    # implementing Metropolis-Hasting Algorithm
    normRnd = np.random.normal(size=nSamples)
    unifRnd = np.random.uniform(size=nSamples)
    xArray = np.zeros(nSamples)
    alphaArray = np.zeros(nSamples)
    for i in range(nSamples):
        previousX = startX if i==0 else xArray[i-1]
        xArray[i] = normRnd[i] + previousX
        alphaArray[i] = min(1., distfunc(xArray[i])/distfunc(previousX))
        xArray[i] = xArray[i] if alphaArray[i]>unifRnd[i] else previousX
    return xArray