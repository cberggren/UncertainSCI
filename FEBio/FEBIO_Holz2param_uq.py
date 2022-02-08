# %% Library/module imports

import numpy as np
from numpy import array_str
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
from scipy import stats
from datetime import datetime

# point to UncertainSCI
import sys
sys.path.append("C:\\Users\\cbergrgren\\GitHub\\UncertainSCI")

from UncertainSCI.distributions import BetaDistribution,TensorialDistribution,NormalDistribution,UniformDistribution
from UncertainSCI.model_examples import laplace_grid_x, laplace_ode, KLE_exponential_covariance_1d
from UncertainSCI.indexing import TotalDegreeSet
from UncertainSCI.pce import PolynomialChaosExpansion

'''
#########################
Description: Generate set of Holzapfel parameters using bivariate distribution to run as a set of FEBio models
Author: Caleb Berggren
Date: 9-20-21
#########################
'''

plt.close('all')
save = 0 # == 1, save file

# %% Specify input parameter distribution
# From Holzapfel et al., 2005 (https://doi.org/10.1152/ajpheart.00934.2004)
# Inputs
# - c: ground matrix stiffness (mu= c/2, c = 2*mu)
# - k1: stress-like parameter for fibers

dim = 2
# Define distribution across domain
med_u = 1.27
med_uSD = np.array(0.25) # standard deviation of c
med_uCov = np.square(med_uSD) # covariance of c
lim_M = 4 # plotting limit mulitplier
med_u_lim = med_uSD*lim_M

med_k1 = 21.60
med_k1SD = np.array(5) # standard deviation of k1
med_k1Cov = np.square(med_k1SD) # covariance of k1
med_k1_lim = med_k1SD*lim_M

bounds = np.zeros([2,dim])
bounds[:,0] = [med_u-med_u_lim, med_u+med_u_lim]    # Bounds for first parameter
bounds[:,1] = [med_k1-med_k1_lim, med_k1+med_k1_lim]  # Bounds for second parameter

# bounds[:,0] = np.reshape(np.array([c-clim,c+clim]), [2, 1])    # Bounds for first parameter
# bounds[:,1] = np.reshape(np.array([k1-klim,k1+klim]), [2, 1])  # Bounds for second parameter

aa = 8.3
dist = BetaDistribution(alpha=aa,beta=aa,domain=bounds)

mu = dist.mean()
cov = dist.cov()

print("The mean of this distribution is")
print(np.array2string(mu))
print("\nThe covariance matrix of this distribution is")
print(np.array2string(cov))

# Create a grid to plot the density
M = 100
x = np.linspace(bounds[0,0], bounds[1,0], M)
y = np.linspace(bounds[0,1], bounds[1,1], M)

X, Y = np.meshgrid(x, y)
XY = np.vstack([X.flatten(), Y.flatten()]).T

pdf = dist.pdf(XY)

# View individual distributions
x = np.linspace(bounds[0,1], bounds[0,1], 100) # define array of c1 to visualize
y = stats.norm.pdf(x,med_u,med_uSD)
fig, ax = plt.subplots(figsize=(100,100))
ax.plot(x,y)
ax.set_title('c1 distributions')
ax.set_xlabel('c1 [kPa]')
ax.plot(x,y)
ax.legend(['Normal','Beta'])


# View bivariate distribution
pdf = np.reshape(pdf, [M, M])

# %matplotlib qt # for interactive plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, pdf, cmap=cm.coolwarm)
fig.colorbar(surf)
ax.set_xlabel('c1')
ax.set_ylabel('k1')
plt.title('PDF for a bivariate "normal" distribution of material parameters')

plt.show()

Hp = np.array([med_u, med_k1]) # Holzapfel parameters
labels = ['c', 'k1']

# %% Initialize for loop, variables

# for ii in it:
#  #   stiffness_label[ii] = f'{labels[ii]=}'.split('=')[0]
    
#     cov[ii] = np.square(s[ii])
#     dist[ii] = NormalDistribution(mean=Hp[ii],cov=np.array(cov[ii]))

# %% Expressivity setup
# Expressivity determines what order of polynomial to use when emulating
# our model function. This is a tuneable hyper parameter, however UncertainSCI
# also has the cabability to auto determine this value. 
order = 5 # PCE order
index_set = TotalDegreeSet(dim=dim, order=order)

# %% Building the PCE

#  First provide the indicies and distribution
pce = PolynomialChaosExpansion(index_set=index_set, distribution=dist)

# Next generate the samples that you want to query
pce.generate_samples()
samples = pce.samples # store samples we will query
samples = samples.round(decimals=2)
print('Parameters to query:')
print(samples)

print('\n')
print('This will query the model {0:d} times'.format(pce.samples.shape[0]))

if save == 1:
    # File names
    now = datetime.now() # Get datetime string
    date_time = now.strftime("y%ym%md%dh%Hm%M")
    folder = r'C:\Users\cbergrgren\Box\UQ\Quarter Cylinder'
    # file = 'Quarter_cylinder_' + stiffness_label + '_parameters_v1'
    file = 'Quarter_cylinder_Holz2param_BivarNorm_v1' + '_' + date_time
    f_path = folder + '\\' + file +'.txt' # construct file path
    units = 'kPa'
    
    # Write query set file
    with open(f_path, 'w',newline='') as f:
        # Write header
        dist_label = 'Bivariate Normal Distribution' # NormalDistribution.__name__
        csv.writer(f, delimiter=',').writerow([dist_label])
        csv.writer(f, delimiter=',').writerow(['mu mean = ' + str(med_u) + ' ' + units])
        csv.writer(f, delimiter=',').writerow(['mu standard deviation = ' + str(med_uSD)[1:-1]])
        csv.writer(f, delimiter=',').writerow(['k1 mean= ' + str(med_k1) + ' ' + units])
        csv.writer(f, delimiter=',').writerow(['k1 standard deviation = ' + str(med_k1SD)[1:-1]])
        csv.writer(f, delimiter=',').writerow(['Polynomial order = ' + str(order)])
        csv.writer(f, delimiter=',').writerow(['--------------------------'])
        row_names = 'mu, k1'
        csv.writer(f, delimiter=',').writerow([row_names])
        csv.writer(f, delimiter=',').writerow(['--------------------------'])
        # Write samples to query
        csv.writer(f, delimiter=',').writerows(samples)

# %% Analysis
# model_evaluations = pce.model_output # feed in FEBio output

































