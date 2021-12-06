#Temporal data analysis

import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

import os
os.chdir('KocPythonFall2021/inclass/9TSCS/')

data = pd.read_stata('WCRWreplication.dta')
data.head()
data.columns

#argument of original analysis is that non-violent as opposed to violent protests are more associated with increases in polity scores (level of democracy)
#what problems may arise in this analysis?
#side note: does this theoretically make sense?

#let's look at the data (politychanget1 and nonviol)
from matplotlib.colors import from_levels_and_colors
cmap, _ = from_levels_and_colors([-.5, .5, 1.5], ['red', 'green']) #red for violent, green for nonviolent
plt.scatter(data.eyear, data.politychanget1, c = data.nonviol, cmap = cmap)
#what do we see?
#Looking at the above plot, it seems that non-violent movements are more associated with increases in polity scores - but what can go wrong?

X = pd.DataFrame(data.nonviol)
X['constant'] = 1
simpleMod = sm.OLS(data.politychanget1, X, missing = 'drop').fit()
simpleMod.summary()
#positive effect (but really?)

import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(simpleMod.resid, lags=40 , alpha=0.05)
acf.show()
#no issues, but how is the data ordered?
data.eyear
#let's reorder by time
data = data.sort_values('eyear')
X = pd.DataFrame(data.nonviol)
X['constant'] = 1
simpleMod = sm.OLS(data.politychanget1, X, missing = 'drop').fit()
acf = smt.graphics.plot_acf(simpleMod.resid, lags=40 , alpha=0.05)
acf.show()
#ok, but not balanced (you NEED to remember this)
#if you are dealing with unbalanced data, talk to me
#we will return to this example in later classes, but for now, we can see from the plot that variance is changing over time, there appears to be trends, etc.
#a major problem is that polity has increased over time (perhaps obviously)
#let's (naively) account for this by including a temporal trend (linear, but we will certainly want something more advanced - to come)
X['eyear'] = data.eyear
linTrendMod = sm.OLS(data.politychanget1, X, missing = 'drop').fit()
linTrendMod.summary()
#let's account for increasing variance (heteroskedasticity) with WLS
X = pd.DataFrame(data.nonviol)
X['constant'] = 1
simpleMod = sm.OLS(data.politychanget1, X, missing = 'drop').fit()
dd = X
dd['politychanget1'] = data.politychanget1
dd = dd.dropna()
X = dd[['nonviol', 'constant']]
WLSmod = sm.WLS(dd.politychanget1, X, weights = 1.0 / (simpleMod.resid ** 2)).fit()
WLSmod.summary()
#still holds (but note the strong parametric assumptions)

#this is all pretty hacky - common fixes assume balanced data, but we will need ML (like GPs) to avoid strong assumptions in unbalanced time-series

#TODO: include relevant controls, test for trends, test for stationarity, etc.




