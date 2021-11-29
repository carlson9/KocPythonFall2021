import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

import os
os.chdir('KocPythonFall2021/inclass/6LinMod/')

data = pd.read_stata('TamingGods.dta')

data.head()
data.describe()
data.columns
#lets check if ethnic fractionalization is correlated with religious repression
data = data[['Religion', 'Ethnic']].dropna()
y = data['Religion']
X = pd.DataFrame(data['Ethnic'])
X['constant'] = 1
myFit = sm.OLS(y, X).fit()

#linearity
#The dependent variable (y) is assumed to be a linear function of the independent variables (X, features) specified in the model. The specification must be linear in its parameters. Fitting a linear model to data with non-linear patterns results in serious prediction errors, especially out-of-sample (data not used for training the model).

#To detect nonlinearity one can inspect plots of observed vs. predicted values or residuals vs. predicted values. The desired outcome is that points are symmetrically distributed around a diagonal line in the former plot or around a horizontal line in the latter one. In both cases with a roughly constant variance.

#Observing a ‘bowed’ pattern indicates that the model makes systematic errors whenever it is making unusually large or small predictions. When the model contains many features, nonlinearity can also be revealed by systematic patterns in plots of the residuals vs. individual features.
def linearity_test(model, y):
    '''
    Function for visually inspecting the assumption of linearity in a linear regression model.
    It plots observed vs. predicted values and residuals vs. predicted values.
    
    Args:
    * model - fitted OLS model from statsmodels
    * y - observed values
    '''
    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1,2)
    
    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    
linearity_test(myFit, y)

#expectation of resids is 0 (obviously it is)
myFit.resid.mean()

#no multicollinearity
#variance inflation factor (VIF)
#the square root of a given variable’s VIF shows how much larger the standard error is, compared with what it would be if that predictor were uncorrelated with the other features in the model. If no features are correlated, then all values for VIF will be 1
#To deal with multicollinearity we should iteratively remove features with high values of VIF. A rule of thumb for removal could be VIF larger than 10 (5 is also common). Another possible solution is to use PCA to reduce features to a smaller set of uncorrelated components.
from statsmodels.stats.outliers_influence import variance_inflation_factor
#we'll need a new model with more covariates
data = pd.read_stata('TamingGods.dta')
data.columns
#lets check if ethnic fractionalization is correlated with religious repression
data = data[['Religion', 'Ethnic', 'polity2_', 'conflict', 'relconflict']].dropna()
y = data['Religion']
X = pd.DataFrame(data[['Ethnic', 'polity2_', 'conflict', 'relconflict']])
X['constant'] = 1
myFit = sm.OLS(y, X).fit()
myFit.summary()
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif[:-1] #no issues

#homoskedasticity
#When residuals do not have constant variance (they exhibit heteroscedasticity), it is difficult to determine the true standard deviation of the forecast errors, usually resulting in confidence intervals that are too wide/narrow. For example, if the variance of the residuals is increasing over time, confidence intervals for out-of-sample predictions will be unrealistically narrow. Another effect of heteroscedasticity might also be putting too much weight to a subset of data when estimating coefficients — the subset in which the error variance was largest.

#To investigate if the residuals are homoscedastic, we can look at a plot of residuals (or standardized residuals) vs. predicted (fitted) values. What should alarm us is the case when the residuals grow either as a function of predicted value or time (in case of time series).

#We can also use two statistical tests: Breusch-Pagan and Goldfeld-Quandt. In both of them, the null hypothesis assumes homoscedasticity and a p-value below a certain level (like 0.05) indicates we should reject the null in favor of heteroscedasticity.

#In the snippets below I plot residuals (and standardized ones) vs. fitted values and carry out the two mentioned tests. To identify homoscedasticity in the plots, the placement of the points should be random and no pattern (increase/decrease in values of residuals) should be visible — the red line in the plots should be flat.
import statsmodels.stats.api as sms
def homoscedasticity_test(model):
    '''
    Function for testing the homoscedasticity of residuals in a linear regression model.
    It plots residuals and standardized residuals vs. fitted values and runs Breusch-Pagan and Goldfeld-Quandt tests.
    
    Args:
    * model - fitted OLS model from statsmodels
    '''
    fitted_vals = model.predict()
    resids = model.resid
    resids_standardized = model.get_influence().resid_studentized_internal

    fig, ax = plt.subplots(1,2)

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Residuals vs Fitted', fontsize=16)
    ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

    sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Scale-Location', fontsize=16)
    ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')

    bp_test = pd.DataFrame(sms.het_breuschpagan(resids, model.model.exog), 
                           columns=['value'],
                           index=['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'])

    gq_test = pd.DataFrame(sms.het_goldfeldquandt(resids, model.model.exog)[:-1],
                           columns=['value'],
                           index=['F statistic', 'p-value'])

    print('\n Breusch-Pagan test ----')
    print(bp_test)
    print('\n Goldfeld-Quandt test ----')
    print(gq_test)
    print('\n Residuals plots ----')

homoscedasticity_test(myFit) #very problematic
#Potential solutions:
#    transformation of the dependent variable
#    in case of time series, deflating a series if it concerns monetary value
#    using ARCH (auto-regressive conditional heteroscedasticity) models to model the error variance. An example might be stock market, where data can exhibit periods of increased or decreased volatility over time


#No autocorrelation of residuals

#This assumption is especially dangerous in time-series models, where serial correlation in the residuals implies that there is room for improvement in the model. Extreme serial correlation is often a sign of a badly misspecified model. Another reason for serial correlation in the residuals could be a violation of the linearity assumption or due to bias that is explainable by omitted variables (interaction terms or dummy variables for identifiable conditions). An example of the former case might be fitting a (straight) line to data, which exhibits exponential growth over time.

#This assumption also has meaning in the case of non-time-series models. If residuals always have the same sign under particular conditions, it means that the model systematically underpredicts/overpredicts what happens when the predictors have a particular configuration.

#To investigate if autocorrelation is present, I use ACF (autocorrelation function) plots and Durbin-Watson test.

#In the former case, we want to see if the value of ACF is significant for any lag (in case of no time-series data, the row number is used). While calling the function, we indicate the significance level (see this article for more details) we are interested in and the critical area is plotted on the graph. Significant correlations lie outside of that area.

#Note: when dealing with data without the time dimension, we can alternatively plot the residuals vs. the row number. In such cases, rows should be sorted in a way that (only) depends on the values of the feature(s).

#The second approach is using the Durbin-Watson test. I do not go into detail how it is constructed but provide a high-level overview. The test statistic provides a test for significant residual autocorrelation at lag 1. The DW statistic is approximately equal to 2(1-a), where a is the lag 1 residual autocorrelation. The DW test statistic is located in the default summary output of statsmodels’s regression.

#Some notes on the Durbin-Watson test:

#    the test statistic always has a value between 0 and 4
#    value of 2 means that there is no autocorrelation in the sample
#    values < 2 indicate positive autocorrelation, values > 2 negative one.
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(myFit.resid, lags=40 , alpha=0.05)
acf.show()
#Potential solutions:

#    in case of minor positive autocorrelation, there might be some room for fine-tuning the model, for example, adding lags of the dependent/independent variables
#    some seasonal components might not be captured by the model, account for them using dummy variables or seasonally adjust the variables
#    if DW < 1 it might indicate a possible problem in model specification, consider stationarizing time-series variables by differencing, logging, and/or deflating (in case of monetary values)
#    in case of significant negative correlation, some of the variables might have been overdifferenced
#    use Generalized Least Squares
#    include a linear (trend) term in case of a consistent increasing/decreasing pattern in the residuals


#The features and residuals are uncorrelated

#To investigate this assumption I check the Pearson correlation coefficient between each feature and the residuals. Then report the p-value for testing the lack of correlation between the two considered series.
from scipy.stats.stats import pearsonr

for column in X.columns[:-1]:
    corr_test = pearsonr(X[column], myFit.resid)
    print(f'Variable: {column} --- correlation: {corr_test[0]:.4f}, p-value: {corr_test[1]:.4f}')
#I cannot reject the null hypothesis (lack of correlation) for any pair.


#variability in X
X.apply(np.var, axis=0)


#Normality of residuals

#When this assumption is violated, it causes problems with calculating confidence intervals and various significance tests for coefficients. When the error distribution significantly departs from Gaussian, confidence intervals may be too wide or too narrow.

#Some of the potential reasons causing non-normal residuals:

#    presence of a few large outliers in data
#    there might be some other problems (violations) with the model assumptions
#    another, better model specification might be better suited for this problem

#Technically, we can omit this assumption if we assume instead that the model equation is correct and our goal is to estimate the coefficients and generate predictions (in the sense of minimizing mean squared error).

#However, normally we are interested in making valid inferences from the model or estimating the probability that a given prediction error will exceed some threshold in a particular direction. To do so, the assumption about the normality of residuals must be satisfied.

#To investigate this assumption we can look at:

#    QQ plots of the residuals (a detailed description can be found here). For example, a bow-shaped pattern of deviations from the diagonal implies that the residuals have excessive skewness (i.e., the distribution is not symmetrical, with too many large residuals in one direction). The s-shaped pattern of deviations implies excessive kurtosis of the residuals — there are either too many or two few large errors in both directions.
#    use statistical tests such as the Kolmogorov-Smirnov test, the Shapiro-Wilk test, the Jarque-Bera test, and the Anderson-Darling test
from scipy import stats

def normality_of_residuals_test(model):
    '''
    Function for drawing the normal QQ-plot of the residuals and running 4 statistical tests to 
    investigate the normality of residuals.
    
    Arg:
    * model - fitted OLS models from statsmodels
    '''
    sm.ProbPlot(model.resid).qqplot(line='s');
    plt.title('Q-Q plot');

    jb = stats.jarque_bera(model.resid)
    sw = stats.shapiro(model.resid)
    ad = stats.anderson(model.resid, dist='norm')
    ks = stats.kstest(model.resid, 'norm')
    
    print(f'Jarque-Bera test ---- statistic: {jb[0]:.4f}, p-value: {jb[1]}')
    print(f'Shapiro-Wilk test ---- statistic: {sw[0]:.4f}, p-value: {sw[1]:.4f}')
    print(f'Kolmogorov-Smirnov test ---- statistic: {ks.statistic:.4f}, p-value: {ks.pvalue:.4f}')
    print(f'Anderson-Darling test ---- statistic: {ad.statistic:.4f}, 5% critical value: {ad.critical_values[2]:.4f}')
    print('If the returned AD statistic is larger than the critical value, then for the 5% significance level, the null hypothesis that the data come from the Normal distribution should be rejected. ')
    
normality_of_residuals_test(myFit)
#Potential solutions:

#    nonlinear transformation of target variable or features
#    remove/treat potential outliers
#    it can happen that there are two or more subsets of the data having different statistical properties, in which case separate models might be considered


#lack of outliers is not an assumption, but outliers may change results, and including/removing/treating needs to be considered in a theoretical fashion
outMat = myFit.outlier_test()
any(outMat['bonf(p)'] < .05)  


