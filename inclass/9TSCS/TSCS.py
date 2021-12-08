import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import os
os.chdir('KocPythonFall2021/inclass/9TSCS/')

data = pd.read_stata('WCRWreplication.dta')
data.head()
data.columns

#argument of original analysis is that non-violent as opposed to violent protests are more associated with increases in polity scores (level of democracy)
#what problems may arise in this analysis?
#side note: does this theoretically make sense?

# TSCS and Panel Data Analyses

#- There are no agreed upon defining characteristics between TSCS and panel data, but for our purposes we will refer to large N, mostly balanced data as panel
#- It is very important to note which of the modeling strategies (most of them) assume balance; this is often overlooked
#- We will first deal with protest data, and the relationship between violent protest and polity change

## One- and Two-Way Fixed Effects Models

#- A `fixed effect' is colloquially simply an indicator, a dummy variable - but really they are all effects that are not assumed randomly distributed
#- In panel data, we can include a `wave' (temporal) effect and/or a unit effect
#- Most common to use two-way (both temporal and unit) for panel data
#- For TSCS, two-way is also common, but the interpretability and framework are hurt by multiple observations for a unit-time
#- Fixed effects control for heterogeneity in the data
#- The inclusion of unit and time fixed effects accounts for both unit-specific (but time-invariant) and time-specific (but unit-invariant) unobserved confounders in a flexible manner
#- Additivity and separability of the two types of unobserved confounders
#- However, their statistical power can be very low with small N, because of the loss in degrees of freedom
#- Unlike random/mixed effects, they do not have distributional assumptions
#- These two points mean that there is no information `borrowed' from other unit-times, and the variation that would be explained by your independent variable is largely absorbed
#- These have great asymptotic properties, but a lot of research on the actual practice show they perform rather poorly, even when assumptions are met
#  + Linearity (although one could nonparametrically adjust for unit-specific (time-specific) unobserved confounders by matching a treated observation with control observations of the same unit (time period), no other observation shares the same unit and time indices. Thus, the 2FE estimator critically relies upon the linearity assumption for its simultaneous adjustment for the two types of unobserved confounders)
#  + Separability (precisely when their respective convex hulls are disjoint (colloquially, do not overlap); means that the marginal rate of substitution between any pair of primary inputs is independent of the amount of intermediate inputs used)
#  + Additivity (means that the effect of one independent variable(s) on the dependent variable does NOT depend on the value of another independent variable(s))
#  + Functional form assumption (are we modeling the DGP?)
#  + Adjustment for the two types of unobserved confounders cannot be done nonparametrically under the 2FE framework
#- Very simple to run; just a linear model with factors
#- Let us explore the models from temporal, but take into account that this is actually (highly unbalanced) TSCS data

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

#lets add a temporal effect - the beginning year seems most appropriate
timeMod = smf.ols('politychanget1 ~ nonviol + C(byear)', data = data).fit()
timeMod.summary() #holds up

#now let's add a location fixed effect
twoWayMod = smf.ols('politychanget1 ~ nonviol + C(byear) + C(location)', data = data).fit()
twoWayMod.summary() #holds up, but keep in mind the very strong assumptions, especially when the fixed effects within country are not allowed to vary over time
#One detail here is that for fixed effect models, the standard errors need to be clustered. So, instead of doing all our estimation by hand (which is only nice for pedagogical reasons), we can use the library linearmodels and set the argument cluster_entity to True
from linearmodels.panel import PanelOLS
mod = PanelOLS.from_formula("politychanget1 ~ nonviol",
                            data=data.set_index(["location", "byear"]))

result = mod.fit(cov_type='clustered', cluster_entity=True)
result.summary
#add FE
mod = PanelOLS.from_formula("politychanget1 ~ nonviol + EntityEffects + TimeEffects",
                            data=data.set_index(["location", "byear"]))

result = mod.fit(cov_type='clustered', cluster_entity=True)
result.summary

#now lets look at a balanced example, which makes more intuitive and mathematical sense
#Bosnian ethnic voting as a function of violence experienced during the war
data2 = pd.read_csv('bosnia.csv')
Log_CasualtyMod = smf.ols('Ethnic_Vote_Share ~ Log_Casualty', data = data2).fit()
Log_CasualtyMod.summary() #notice the lack of reliable effects
#lets add a province dummy
Log_CasualtyMod2 = smf.ols('Ethnic_Vote_Share ~ Log_Casualty + C(Municipality)', data = data2).fit()
Log_CasualtyMod2.summary() #notice the lack of reliable effects
#add temporal effects
Log_CasualtyMod3 = smf.ols('Ethnic_Vote_Share ~ Log_Casualty + C(Municipality) + C(Year)', data = data2).fit()
Log_CasualtyMod3.summary() #notice the lack of reliable effects
Log_CasualtyMod.params['Log_Casualty']
Log_CasualtyMod2.params['Log_Casualty']
Log_CasualtyMod3.params['Log_Casualty']
#the coefficient gets larger, even though there are very few degrees of freedom - why?
#we are no longer strictly controlling for heterogeneity, but exploring/exploiting province-year-level variation
#cluster
mod = PanelOLS.from_formula("Ethnic_Vote_Share ~ Log_Casualty + EntityEffects + TimeEffects", data=data2.set_index(["Municipality", "Year"]))

result = mod.fit(cov_type='clustered', cluster_entity=True)
result.summary #errors, why?
#Log_Casualty is constant within Municipalities, but Ethnic_Vote_Share is not - this module should be able to estimate this model, but it cannot
#instead:
Log_CasualtyMod4 = smf.ols('Ethnic_Vote_Share ~ Log_Casualty + C(Municipality) + C(Year)', data = data2).fit(cov_type = 'cluster', cov_kwds = {'groups': data2.Municipality})
Log_CasualtyMod4.summary() #now we have effects - clustered ses DO NOT ALWAYS WIDEN CIs!


#this model is strange (one year is pre-treatment)
#this is an appropriate dataset for introducing the Difference-in-Difference (DiD) estimator
#- DiD estimators have traditionally been used to analyze the effect of a treatment at different times
#- There is absolutely no reason not to use them for continuous treatments though
#- Basically, we interact temporal fixed effects (but not the first, or baseline year) with the variable of interest (and include the constituent terms of time), and include unit fixed effects
#- This is an ideal model for the above Bosnian data
#- However, we need to assume parallel trends, i.e. the units (municipalities) have parallel trends in the outcome if it were not for treatment

#create a dummy for 2006, 2010, and 2014
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
label_binarizer_output = label_binarizer.fit_transform(data2.Year)[:,1:] #we do not want 1990 in our dummy
Log_CasualtyModDiD = smf.ols('Ethnic_Vote_Share ~ Log_Casualty*label_binarizer_output + C(Municipality)', data = data2).fit(cov_type = 'cluster', cov_kwds = {'groups': data2.Municipality, 'time': data2.Year})
Log_CasualtyModDiD.summary() #we care about the interaction effects
#what happens if we do not use clustered ses?
Log_CasualtyModDiD = smf.ols('Ethnic_Vote_Share ~ Log_Casualty*label_binarizer_output + C(Municipality)', data = data2).fit()
Log_CasualtyModDiD.summary()
#what if we use robust standard errors (NEVER DO THIS!... unless a reviewer asks)
Log_CasualtyModDiD = smf.ols('Ethnic_Vote_Share ~ Log_Casualty*label_binarizer_output + C(Municipality)', data = data2).fit()
Log_CasualtyModDiD.get_robustcov_results(cov_type='HC1').summary()
#the results are MORE significant

#visualizing fixed effects
toy_panel = pd.DataFrame({
    "mkt_costs":[5,4,3.5,3, 10,9.5,9,8, 4,3,2,1, 8,7,6,4],
    "purchase":[12,9,7.5,7, 9,7,6.5,5, 15,14.5,14,13, 11,9.5,8,5],
    "city":["C0","C0","C0","C0", "C2","C2","C2","C2", "C1","C1","C1","C1", "C3","C3","C3","C3"]
})

m = smf.ols("purchase ~ mkt_costs", data=toy_panel).fit()

plt.scatter(toy_panel.mkt_costs, toy_panel.purchase)
plt.plot(toy_panel.mkt_costs, m.fittedvalues, c="C5", label="Regression Line")
plt.xlabel("Marketing Costs (in 1000)")
plt.ylabel("In-app Purchase (in 1000)")
plt.title("Simple OLS Model")
plt.legend();

fe = smf.ols("purchase ~ mkt_costs + C(city)", data=toy_panel).fit()

fe_toy = toy_panel.assign(y_hat = fe.fittedvalues)

plt.scatter(toy_panel.mkt_costs, toy_panel.purchase, c=toy_panel.city)
for city in fe_toy["city"].unique():
    plot_df = fe_toy.query(f"city=='{city}'")
    plt.plot(plot_df.mkt_costs, plot_df.y_hat, c="C5")

plt.title("Fixed Effect Model")
plt.xlabel("Marketing Costs (in 1000)")
plt.ylabel("In-app Purchase (in 1000)");



## Mixed (Random and Fixed) Effects

#- We can use these in much the same way (and a lot more) as FEs
#- Random effects have a distributional assumption
#- This distribution buys us a lot of things
#  + We can borrow information from other units to find a generalized pattern across units
#  + The statistical power is much higher
#  + The power means we can get false positives, but generally only under mis-specification
#  + We can directly estimate the variation of the underlying distribution, meaning we can inspect which levels need variation accounted for
#  + We no longer make the assumption of separability, and estimate the covariation of the parameters
#- The distribution, in general, assumes that the population under study is representative of the group-of-interest, in that the effects are random realizations of the randomly distributed population (and temporal) heterogeneity (prevalancy kind of)
#- We will generally want to go Bayesian, for a number of reasons we will cover later, but basically it allows for much more control over the model and a much more flexible DGP
#- Random effects do not require balance
#- You can model units that only have one observation
#- You can model a small number of units
#- You can model a small number of observations
#- You can add predictors to the random effects for more precision (contextual effects)
#- You can also model any slope coefficient randomly, allowing it to vary at any desired level
#- Assumes that errors are uncorrelated with regressors
#- Growth curves of pigs
data = sm.datasets.get_rdataset("dietox", "geepack").data
md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
mdf = md.fit()
print(mdf.summary()) #notice the group variation - this indicates whether or not the random effect is warranted

#Note: MLE estimates of RE models are common, but you should really go Bayesian for them

## Hausman Test

#- The null hypothesis is that the preferred model is random effects vs. the alternative the fixed effects
#- Tests whether the unique errors are correlated with the regressors, the null hypothesis is they are not
#- Unfortunately this is a terrible test, but there are not great tests - REs are always my preference, but you may have to please reviewers with FEs - choice should be theoretical in nature

#TODO: look at the documentation for smf.mixedlm - figure out how to let the slopes vary by groups, and apply it some WB data (e.g., effect of trade on GDP, allowing the effect to vary by country)





