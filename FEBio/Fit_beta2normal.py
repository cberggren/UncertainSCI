# %% Library/module imports

import numpy as np
from numpy import array_str
from matplotlib import pyplot as plt
import csv
from scipy import stats

# point to UncertainSCI
import sys
sys.path.append("C:\\Users\\cbergrgren\\GitHub\\UncertainSCI")

from UncertainSCI.distributions import BetaDistribution,UniformDistribution,NormalDistribution
from UncertainSCI.model_examples import laplace_grid_x, laplace_ode, KLE_exponential_covariance_1d
from UncertainSCI.indexing import TotalDegreeSet
from UncertainSCI.pce import PolynomialChaosExpansion

'''
#########################
Description: Fit beta distribution to a normal distribution for given set of parameters
Author: Caleb Berggren
Date: 9-28-21
Version changes: N/A
#########################
'''

# %% Specify input parameter distribution
# Inputs: shear stiffness (c1). Note, c1 typically = u/2, but in Holzapfel et al., 2005,
# stiffness of ground matrix presented as u (no division by 2)

# neo-Hookean Model has 1 input parameter, so dimensionality = 1
dimension = 1

# Define distribution across domain
m = 1.27 # total stiffness
stiffness_label = f'{m=}'.split('=')[0]
s = np.array([0.63]) # standard deviation
cov = np.square(s)
dist = NormalDistribution(mean=m,cov=cov)
dist_label = NormalDistribution.__name__

# Create beta distribution to match
a = 1 # plotting limit mulitplier
lim = s*a
bounds = np.reshape(np.array([m-lim,m+lim]), [2, 1])
aa = 5
distB = BetaDistribution(alpha=aa,beta=aa,domain=bounds)
distN = NormalDistribution(mean=m,cov=cov)

x = np.linspace(bounds[0,0], bounds[1,0], 100) # define array of c1 to visualize
X = np.meshgrid(x)
X = np.vstack([X]).T
pdf = distB.pdf(X)

# View distribution
y = stats.norm.pdf(x,m,s)
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(x,y)
# ax.set_title('distributions')
# ax.set_xlabel('c1 [kPa]')
ax.plot(x,pdf)
ax.legend(['Normal','Beta'])

































