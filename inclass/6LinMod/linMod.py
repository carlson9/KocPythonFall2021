import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats
import statsmodels.api as sm

import os
os.chdir('KocPythonFall2021/inclass/6LinMod/')

data = pd.read_stata('TamingGods.dta')

data.head()
data.describe()
data.columns
#there is a pdf with explanations of variables
#lets check if ethnic fractionalization is correlated with religious repression
y = data['Religion']
X = pd.DataFrame(data['Ethnic'])
X['constant'] = 1
myFit = sm.OLS(y, X, missing = 'drop').fit()
myFit.summary()
myFit.conf_int()

#now lets code our own MLE algorithm
#rather than maximize LL, minimize neg LL
# define likelihood function
def MLERegression(params):
    intercept, beta, sd = params[0], params[1], params[2] # inputs are guesses at our parameters
    yhat = intercept + beta*x # predictions# next, we flip the Bayesian question
    # compute PDF of observed values normally distributed around mean (yhat) with a standard deviation of sd
    negLL = -np.sum( stats.norm.logpdf(y, loc=yhat, scale=sd) )# return negative LL
    return(negLL)

# let’s start with some random coefficient guesses and optimize
dd = data[['Religion', 'Ethnic']].dropna()
y = dd['Religion']
x = dd['Ethnic']
guess = np.array([.5,.5,2])
results = minimize(MLERegression, guess, method = 'Nelder-Mead', options={'disp': True, 'maxiter': 1000})
results
myFit.params
sns.regplot(x,y)

#TODO: add some PROPER controls, reassess

#TODO: write a function that splits the dataset into training (.8 prop) and test (.2 prop), runs a model on the training, predicts using the test, and calculates RMSE between predicted y and true y - repeat and average



