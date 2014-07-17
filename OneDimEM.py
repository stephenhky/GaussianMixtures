__author__ = 'hok1'

import numpy as np
from scipy.stats import norm
import OneDimGaussianMixtures as gauss1d
from operator import and_
from functools import partial

class ExpectationMaximizationWorker:
    def EStep(self, xarray, parameters):
        pdf_values = gauss1d.density(xarray, parameters)
        tau = np.zeros([len(parameters), len(xarray)])
        for i in range(len(parameters)):
            zarray = (xarray-parameters[i]['mean'])/parameters[i]['stdev']
            for j in range(len(xarray)):
                tau[i, :] = parameters[i]['weight']*norm.pdf(zarray)/pdf_values[:]
        return tau

    def Mstep(self, xarray, parameters, tau):
        pi = np.sum(tau, axis=1) / len(xarray)
        mu = np.zeros(len(parameters))
        Sigma = np.zeros(len(parameters))
        sum_tau = np.sum(tau, axis=1)
        for i in range(len(parameters)):
            mu[i] = np.sum(tau[i, :]*xarray) / sum_tau[i]
            Sigma[i] = np.sum(tau[i,:]*((xarray-mu[i])**2)) / sum_tau[i]
        new_parameters = map(lambda weight, mean, stdev: {'weight': weight, 'mean': mean, 'stdev': stdev},
                             pi, mu, Sigma)
        return new_parameters

    def workflow(self, xarray, init_parameters, max_iter=1000, tol=1e-4, weight_lowerthreshold=1e-8):
        parameters = init_parameters
        for iterid in range(max_iter):
            tau = self.EStep(xarray, parameters)
            new_parameters = self.Mstep(xarray, parameters, tau)
            if self.parameters_close(parameters, new_parameters, tol=tol):
                return new_parameters
            new_parameters = filter(lambda component: component['weight'] > weight_lowerthreshold, new_parameters)
            if 0 in map(lambda component: component['stdev'], new_parameters):
                return new_parameters
            parameters = new_parameters
        return parameters

    def parameters_close(self, param1, param2, tol=1e-4):
        if len(param1) != len(param2):
            return False
        for comp1, comp2 in zip(sorted(param1, key=lambda comp: comp['weight']), sorted(param2, key=lambda comp: comp['weight'])):
            if not reduce(and_, map(lambda param: abs(comp1[param]-comp2[param])<tol, comp1.keys())):
                return False
        return True

if __name__ == '__main__':
    print 'Sampling... '
    xarray = gauss1d.MetropolisMarkovChain(partial(gauss1d.density, parameters=gauss1d.sample_parameters), 5000)
    init_parameters = [{'weight': 0.5, 'mean': 0, 'stdev': 2},
                       {'weight': 0.5, 'mean': 5, 'stdev': 2}]
    print 'Expectation maximization... '
    em = ExpectationMaximizationWorker()
    parameters = em.workflow(xarray, init_parameters)
    print parameters