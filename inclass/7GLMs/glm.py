import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

import os
os.chdir('KocPythonFall2021/inclass/7GLMs/')

#below will error, because of the version of stata (you may run into this)
#data = pd.read_stata('repdata.dta')
import pyreadstat
data, meta = pyreadstat.read_dta('repdata.dta')
data.columns

#binary outcome

#lets predict war as a function of Oil
#controls: "empgdpenl"  "emplpopl" "empolity2l" (notice the lags)

data['constant'] = 1
#logit (most common)
modLogit = sm.Logit(data.war, data[['constant', 'Oil', "empgdpenl", "emplpopl", "empolity2l"]], missing = 'drop').fit()
modLogit.summary()

#probit
modProbit = sm.Probit(data.war, data[['constant', 'Oil', "empgdpenl", "emplpopl", "empolity2l"]], missing = 'drop').fit()
modProbit.summary() #notice the different scales

#c-log-log (very uncommon)
modCLL = sm.GLM(data.war, data[['constant', 'Oil', "empgdpenl", "emplpopl", "empolity2l"]], missing = 'drop', family = sm.families.Binomial(sm.families.links.cloglog)).fit()
modCLL.summary()


#counts

#lets predict number of wars as a function of democracy

#Poisson
modPois = sm.Poisson(data.wars, data[['constant', 'empolity2l', 'empgdpenl', 'emplpopl', 'ethfrac', 'relfrac']], missing = 'drop').fit()
modPois.summary()

#negative binomial (for overdispersed)
modNB = sm.NegativeBinomial(data.wars, data[['constant', 'empolity2l', 'empgdpenl', 'emplpopl', 'ethfrac', 'relfrac']], missing = 'drop').fit() #notice the warning
modNB = sm.NegativeBinomial(data.wars, data[['constant', 'empolity2l', 'empgdpenl', 'emplpopl', 'ethfrac', 'relfrac']], missing = 'drop').fit(maxiter = 1000) #still - cannot trust results from NB in this example


#multiple categories

#lets predict region by ethnic frac

#logistic multinomial
import statsmodels.discrete as smd
modMN = smd.discrete_model.MNLogit(data.region, data[['constant', 'ethfrac', 'gdptype']], missing = 'drop').fit()
modMN.summary()

#ordered logistic regression
#just for illustration, we'll repeat the number of wars, but this is inappropriate because it is theoretically unbounded
#need dev version of statsmodels
#pip install git+https://github.com/statsmodels/statsmodels
from statsmodels.miscmodels.ordinal_model import OrderedModel
modOL = OrderedModel(data.wars, data[['constant', 'empolity2l', 'empgdpenl', 'emplpopl', 'ethfrac', 'relfrac']], missing = 'drop', distr = 'logit').fit()

#quasibinomial regression for proportions
#predict ethnic frac as a function of polity
modQB = sm.GLM(data.ethfrac, data[['constant', 'empolity2l']], family = sm.families.Binomial(), missing = 'drop').fit()

#TODO: look through https://www.statsmodels.org/stable/examples/index.html and run an atypical glm (e.g., zero-inflated for wars)


