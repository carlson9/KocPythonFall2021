import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import random

import os
os.chdir('KocPythonFall2021/inclass/6LinMod/')

data = pd.read_stata('TamingGods.dta')

#there is a pdf with explanations of variables
#lets check if ethnic fractionalization is correlated with religious repression
data = data[['Religion', 'Ethnic', 'polity2_', 'conflict', 'relconflict']]
y = data[['Religion']]
X = pd.DataFrame(data[['Ethnic', 'polity2_', 'conflict', 'relconflict']])
X['constant'] = 1
myFit = sm.OLS(y, X, missing = 'drop').fit()
myFit.summary()
#recall the errors are highly problematic

#bootstrap!
#This is an important concept to understand. Bootstrapping is a general approach to statistical inference based on building a sampling distribution for a statistic by resampling from the data at hand. Bootstrapping offers advantages:
#- The bootstrap is quite general, although there are some cases in which it fails (best example: if there is structure, you will do better modeling the structure, such as temporal correlation, cross-sectional dependencies, etc.; the bootstrap will still generally produce unbiased but inefficient results, and aggregation is not always a good depiction of heterogeneous reality).
#- Because it does not require distributional assumptions (such as normally distributed errors), the bootstrap can provide more accurate inferences when the data are not well behaved or when the sample size is small.
#- It is possible to apply the bootstrap to statistics with sampling distributions that are difficult to derive, even asymptotically.
#- It  is  relatively  simple  to  apply  the  bootstrap  to  complex  data-collection plans (such as stratified and clustered samples).
#- We can even use it to get uncertainty for models (such as NN) that do not lend to uncertainty (teaser for ML)

#algorithm (dataset has N observations):
#    1) sample N rows / observations WITH REPLACEMENT
#    2) run your model
#    3) collect the estimates' coefficients
#    4) repeat 1-3 several times (about 1000 is usually sufficient)
#    5) take the mean of the (many) estimates -> point estimate
#    6) find the std. dev. of the (many) point estimates -> sd of coefficient
#    7) can also use empirical quantiles for conf ints

def sampleMod(y, X, seed = 1):
    random.seed(seed)
    data = pd.DataFrame(np.hstack((y, X))).dropna()
    data = data.sample(n = data.shape[0], replace = True)
    return(sm.OLS(data[0], data.iloc[:, 1:]).fit().params)

def bootstrap(y, X, iters = 1000):
    pars = np.empty((X.shape[1], iters), dtype = float)
    for i in range(iters): pars[:, i] = sampleMod(y, X, i)
    pars = pd.DataFrame(pars)
    betas = pars.apply(np.mean, axis = 1)
    stddevs = pars.apply(np.std, axis = 1)
    CIs = np.quantile(pars, [.025, .975], axis = 1).T
    summaryOut = pd.DataFrame(np.column_stack((np.reshape(betas.values, (len(betas), 1)),
np.reshape(stddevs.values, (len(stddevs), 1)),
CIs)),
columns = ['Estimates', 'Std.Dev.', '0.025', '0.975'])
    return(summaryOut)

def compareBootOLS(y, X, iters = 1000):
    myFit = sm.OLS(y, X, missing = 'drop').fit()
    EstimatesOLS = myFit.params
    StdDevOLS = myFit.bse
    ConfIntOLS = myFit.conf_int()
    summaryOLS = pd.DataFrame(np.column_stack((EstimatesOLS, StdDevOLS, ConfIntOLS)), columns = ['EstimatesOLS', 'Std.Dev.OLS', '0.025OLS', '0.975OLS'])
    summaryBoot = bootstrap(y, X, iters)
    compOut = pd.concat([summaryOLS, summaryBoot], axis = 1)
    compOut = compOut.set_index(ConfIntOLS.index)
    return(compOut)

compareBootOLS(y, X)

