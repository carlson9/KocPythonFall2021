import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

import os
os.chdir('KocPythonFall2021/inclass/7GLMs/')

final_df = pd.read_csv('data.csv')
final_df.columns
final_df['constant'] = 1

pop_logit = sm.Logit(final_df.intercon, final_df[['constant', 'country_pop', 'aggdifxx', 'gdppc', 'polity2']], missing = 'drop').fit()
pop_logit.summary()

#let's get a LaTeX table
print(pop_logit.summary().as_latex())
#now a csv
pop_logit.summary().as_csv()
#html
pop_logit.summary().as_html()
#text
pop_logit.summary().as_text()

#exponentiate the coefficients (why?)
params = pop_logit.params
conf = pop_logit.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))

#coefficient plot
points = pop_logit.params
lower = pop_logit.params - pop_logit.conf_int()[0]
upper = pop_logit.conf_int()[1] - pop_logit.params
yerr = np.row_stack((lower, upper))

fig, ax = plt.subplots(figsize=(8, 5))
plt.errorbar(x = np.arange(points.shape[0]), y = points, yerr = yerr, ls = 'none')
ax.set_ylabel('')
ax.set_xlabel('')
ax.scatter(x=np.arange(points.shape[0]), 
           marker='s', s=35, 
           y=points)
ax.axhline(y=0, linestyle='--', color='black', linewidth=1)
ax.set_xticks(np.arange(points.shape[0]))
ax.set_xticklabels(['Constant', 'Pop.', 'Agg.Diff.', 'GDPPC', 'Polity'])

#plot predictions
X = final_df[['constant', 'country_pop']].sort_values(by = 'country_pop').dropna()
X['aggdifxx'] = np.mean(final_df['aggdifxx'])
X['gdppc'] = np.mean(final_df['gdppc'])
X['polity2'] = np.mean(final_df['polity2'])
preds = pop_logit.predict(exog = X)
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(X['country_pop'], preds)
ax.plot(X['country_pop'], [preds.min() - .05]*len(X['country_pop']), '|', color='k')

#TODO: fit a Poisson of number of wars on democracy with controls (we did this in glm.py). plot expected numbers of wars as a function of democracy as we did above but with your new model

#TODO: replace the logit explanatory variable (country_pop) with groupcon. using a train/test split (run several times) assess which explanatory variable is better at prediction as captured by RMSE



