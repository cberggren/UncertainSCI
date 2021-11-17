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
Description: Generate list of Neo-Hookean parameters to run as a set of FEBio models
Author: Caleb Berggren
Date: 8-12-21
Version changes: View distribution
#########################
'''

save = 0
# %% Specify input parameter distribution
# Inputs: shear stiffness (c1). Note, c1 typically = u/2, but in Holzapfel et al., 2005,
# stiffness of ground matrix presented as u (no division by 2)

# neo-Hookean Model has 1 input parameter, so dimensionality = 1
dimension = 1

# Define distribution across domain
adv_c1 = round(84.7,2) # adventitial stiffness
med_c1 = round(44.47,2) # medial stiffness
c1 = round(adv_c1 + med_c1,2) # total stiffness
stiffness_label = f'{c1=}'.split('=')[0]
s = np.array([5]) # standard deviation
cov = np.square(s)
dist = NormalDistribution(mean=c1,cov=cov)
dist_label = NormalDistribution.__name__

# Create beta distribution to match
a = 4
clim = s*a
bounds = np.reshape(np.array([c1-clim,c1+clim]), [2, 1])
aa = 8.3
distB = BetaDistribution(alpha=aa,beta=aa,domain=bounds)
distN = NormalDistribution(mean=c1,cov=cov)

x = np.linspace(bounds[0,0], bounds[1,0], 100) # define array of c1 to visualize
X = np.meshgrid(x)
X = np.vstack([X]).T
pdf = distB.pdf(X)

# View distribution
y = stats.norm.pdf(x,c1,s)
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(x,y)
ax.set_title('c1 distributions')
ax.set_xlabel('c1 [kPa]')
ax.plot(x,pdf)
ax.legend(['Normal','Beta'])

# %% Expressivity setup
# Expressivity determines what order of polynomial to use when emulating
# our model function. This is a tuneable hyper parameter, however UncertainSCI
# also has the cabability to auto determine this value. 
order = 1
index_set = TotalDegreeSet(dim=dimension, order=order)

# %% Building the PCE

#  First provide the indicies and distribution
pce = PolynomialChaosExpansion(index_set=index_set, distribution=dist)

# Next generate the samples that you want to query
pce.generate_samples()
samples = pce.samples # store samples we will query
samples = samples.round(decimals=2)
print('Stiffnesses to query:')
print(samples)

print('\n')
print('This will query the model {0:d} times'.format(pce.samples.shape[0]))

if save == 1:
    # File names
    folder = r'C:\Users\cbergrgren\Box\UQ\Quarter Cylinder'
    file = 'Quarter_cylinder_' + stiffness_label + '_parameters_v1'
    f_path = folder + '\\' + file +'.txt' # construct file path
    units = 'kPa'
    
    # Write query set file
    with open(f_path, 'w',newline='') as f:
        # Write header
        csv.writer(f, delimiter=',').writerow([dist_label])
        csv.writer(f, delimiter=',').writerow(['Mean stiffness = ' + str(c1) + ' ' + units])
        csv.writer(f, delimiter=',').writerow(['Standard deviation = ' + str(s)[1:-1]])
        csv.writer(f, delimiter=',').writerow(['Polynomial order = ' + str(order)])
        # Write samples to query
        csv.writer(f, delimiter=',').writerows(samples)

# %% Analysis
# model_evaluations = pce.model_output # feed in FEBio output

































