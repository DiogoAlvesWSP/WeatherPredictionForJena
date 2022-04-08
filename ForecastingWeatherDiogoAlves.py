"""
Weather Forecasting problem for the Jena Dataset
Comments are marked with a #. 
If running, please stop in each block and read the comments carefully! Each block ends in a space 
and the next begins with a comment sign.
Dataset: https://www.kaggle.com/pankrzysiu/weather-archive-jena
Description of the variables of the data:
p (mbar) - Air pressure (SI: bar)
T (degC) - Air Temperature (SI: Celsius)
Tpot (K) - Air Temperature (SI: Kelvin (+273.42 K))
rh - relative humidity
VPmax, VPact, VPdef - Vapor pressure (maximum, actual, definite(?))
sh - No idea!
H2OC - Water concentration or humidity
rho - Air density (SI - g/m**3)
wv, maxwv - Wind velocity (average, maximum) (SI - m/s)
wd - Wind direction (SI - Deg)
"""

# Imports, including packages for Regression and Data Visualization, error metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot,autocorrelation_plot
import seaborn as sn
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import pickle
from datetime import datetime
from pmdarima import auto_arima
from scipy.stats import pearsonr
from tbats import TBATS
from autots import AutoTS
import matplotlib.dates as mdates

# Reads the csv file from disk
WeatherTimeSeries = pd.read_csv(
        r'HOMEDIRECTORY\jena_climate_2009_2016.csv', 
        parse_dates=True,
        index_col=0)
WeatherTimeSeries.index.name = 'Date/Time'
WeatherTimeSeries.reset_index(inplace=True)
WeatherSeriesDate = WeatherTimeSeries.iloc[:,0]
WeatherSeriesDate = pd.to_datetime(WeatherSeriesDate)

# Temperature as the Target:
WeatherSeriesTarget = WeatherTimeSeries.iloc[:,2]
# Getting explanatory variables to explain temperature
Explanatory_Vars = WeatherTimeSeries.columns.tolist()
Explanatory_Vars.remove('Date/Time')
Explanatory_Vars.remove('T (degC)')
WeatherSeriesExplanatory = WeatherTimeSeries.loc[:, Explanatory_Vars]
Series = [WeatherSeriesTarget,WeatherSeriesExplanatory]
WeatherSeries = pd.concat(Series,axis=1)

# Basic descriptive statistics of Target, see whether there are outliers, Missing values, extreme values...
WeatherSeriesTarget.describe()
plt.plot(WeatherSeriesTarget)
plt.title("Weather as Temperature Over Time")
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
plt.hist(WeatherSeriesTarget, bins=10)
plt.show()
# A boxplot:
sn.boxplot(WeatherSeriesTarget,orient = "h",palette = "Set2")
# Mean temp of around 10 degrees.

# Describing explanatory variables individually:
for column in WeatherSeriesExplanatory.columns:
    WeatherSeriesExplanatory[column].describe()
    figure = plt.figure
    plt.plot(WeatherSeriesExplanatory[column])
    plt.title(column)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()
# Plots together for viewing convenience:
fig, axes = plt.subplots(nrows=3, ncols=4, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = WeatherSeriesExplanatory[WeatherSeriesExplanatory.columns[i]]
    ax.plot(data, color='blue', linewidth=1)
    # Decorations
    ax.set_title(WeatherSeriesExplanatory.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
plt.tight_layout();
# Most of the variables are seasonal as well, except the last 3. Except for these three, they co-evolve, apparently. 

# See how Target and explanatory variables co-evolve, first by plotting together, 
# then finding correlation, and scatterplots:
for col in WeatherSeriesExplanatory.columns:
    figure = plt.figure
    ax = plt.gca()
    ax.scatter(WeatherSeriesExplanatory[col], WeatherSeriesTarget)
    ax.set_xlabel(col)
    ax.set_ylabel("Temperature")
    ax.set_title("Scatterplot {} and {}".format(col,"Temperature"))
    plt.legend()
    plt.show()
# Views together:       
fig, axes = plt.subplots(nrows=3, ncols=4, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = WeatherSeriesExplanatory[WeatherSeriesExplanatory.columns[i]]
    ax.scatter(data, WeatherSeriesTarget)
    # Decorations
    ax.set_title(WeatherSeriesExplanatory.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_title("{} and {}".format(WeatherSeriesExplanatory.columns[i],"Target"))
    ax.tick_params(labelsize=6)
plt.tight_layout();

# A simple Correlation Matrix to find the most correlated variables:
corr = WeatherSeries.corr()
sn.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
CorrVars = WeatherSeries.corr()["T (degC)"].sort_values(ascending = False).to_frame()
CorrVars.index.name = 'Variable'
CorrVars.reset_index(inplace=True)
CorrVars = CorrVars.iloc[1:]
# Finding most correlated variables:
CorrVars.rename(columns = {'T (degC)':'Correlation'}, inplace = True)
CorrVars = CorrVars.loc[abs(CorrVars['Correlation']) >= 0.5].iloc[:,0].to_dict()
# Subsetting the Explanatory dataframe to these:
Interesting_Explanatory = WeatherSeriesExplanatory[[col_name for col_name in CorrVars.values()]]
# Note how all the variables except the last 3 are relevant. We could easily exclude them from analysis.
# In other words, wind speed and wind direction have no effect on Temperature. 
# Air pressure seems to have a nonlinear relationship, whereas air density has a negative relationship with temperature.

## Repeating the heatmap only with these variables and Temperature:
temp = [WeatherSeriesTarget,Interesting_Explanatory]
temp = pd.concat(temp,axis=1)
corr_2 = temp.corr()
sn.heatmap(corr_2, 
            xticklabels=corr_2.columns.values,
            yticklabels=corr_2.columns.values)

# Plotting the most correlated time series together for Visual inspection:
for column in Interesting_Explanatory.columns:
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    ax_left.plot(WeatherSeriesTarget, color='black')
    ax_right.plot(Interesting_Explanatory[column], color='red')   
    ax.set_title("TimeSeries {} and {}".format("Temperature",column))
    plt.show()
    
# Lag plot and Autocorrelation Function of target:
lag_plot(WeatherSeriesTarget)
autocorrelation_plot(WeatherSeriesTarget)    

# Seasonal and trend decomposition of original series
Decomp = sm.tsa.seasonal_decompose(WeatherSeriesTarget,freq = 365 * 24 * 6)
fig = Decomp.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()
# Note absence of upward or downward trend of temperatures, as well as clear seasonal cycles at several levels (days, months, years) in humps!

# Testing for stationarity of the series through the ADF test:
def adf_test(ts, signif=0.05):
    dftest = adfuller(ts, autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags','# Observations'])
    for key,value in dftest[4].items():
       adf['Critical Value (%s)'%key] = value
    print (adf)    
    p = adf['p-value']
    if p <= signif:
        print(f"Stationary")
    else:
        print(f"Non-Stationary")
        
#apply adf test on the series (Stationarity)
adf_test(WeatherSeriesTarget)
# Doing ADF on each time series:
for name, column in WeatherSeries.iteritems():
    adf_test(column)
    print('\n',column.name)
# The conclusion is that all series are I(0), i.e stationary, as wel could see from visual inspection!

## Choose a level of aggregation (if needed!) Can choose several and proceed:
## Note that there are several cycles within each time period, hourly, daily, etc.
test_3 = [WeatherSeriesDate,WeatherSeriesTarget,WeatherSeriesExplanatory]
Full_explanatory = pd.concat(test_3,axis=1)

# Aggregating Time series by hour, day, week, and month, then choose a level:
Full_explanatory['Date/Time'] = pd.to_datetime(Full_explanatory['Date/Time'])
Hourly_Full_explanatory = Full_explanatory.resample('60T', on='Date/Time').mean()
Hourly_Full_explanatory.index.name = 'Date/Time'
Hourly_Full_explanatory.reset_index(inplace=True)
Hourly_Full_explanatory['Date/Time'] = pd.to_datetime(Hourly_Full_explanatory['Date/Time'])
Daily_Full_explanatory = Full_explanatory.resample('1440T', on='Date/Time').mean()
Daily_Full_explanatory.index.name = 'Date/Time'
Daily_Full_explanatory.reset_index(inplace=True)
Daily_Full_explanatory['Date/Time'] = pd.to_datetime(Daily_Full_explanatory['Date/Time'])
Weekly_Full_explanatory = Full_explanatory.resample('W', on='Date/Time').mean()
Weekly_Full_explanatory.index.name = 'Date/Time'
Weekly_Full_explanatory.reset_index(inplace=True)
Weekly_Full_explanatory['Date/Time'] = pd.to_datetime(Weekly_Full_explanatory['Date/Time'])
Monthly_Full_explanatory = Full_explanatory.resample('M', on='Date/Time').mean()
Monthly_Full_explanatory.index.name = 'Date/Time'
Monthly_Full_explanatory.reset_index(inplace=True)
Monthly_Full_explanatory['Date/Time'] = pd.to_datetime(Monthly_Full_explanatory['Date/Time'])
# I do not have computing power enough to calculate data hourly, can choose, WLOG, daily means:

# Daily Mean temperature looks like a good aggregation level. Look at periodicity:
res = sm.tsa.seasonal_decompose(Daily_Full_explanatory.iloc[:,1].dropna(),freq = 365)
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()
# There are still several seasonal cycles, monthly, and yearly at this level of aggregation.
# The months can clearly be seen as humps on the seasonal graph over a year (around 12)

# This block separates out data for training and testing:
# Use 2016 as test data, noting there is a day of 2017 in the end of the test data (367 data points to predict from test)!
Train_Beginning_index = 0
Train_end_index = len(Daily_Full_explanatory) - len(Daily_Full_explanatory[Daily_Full_explanatory['Date/Time'].dt.strftime('%Y.%m.%d').str.contains("2016")]) - 1
Test_Beginning_index = Train_end_index
Test_end_index = len(Daily_Full_explanatory)
Train_Time_Series = Daily_Full_explanatory.iloc[Train_Beginning_index:Train_end_index,:]
Test_Time_Series = Daily_Full_explanatory.iloc[Test_Beginning_index:Test_end_index,:] # last year for testing

# Testing several approaches on train data:
# Approach 1: AutoML approach TBATS (see https://pypi.org/project/tbats/), uses only data from time series minus covariates 
Train_Target = Train_Time_Series.iloc[:,1]
Test_Target = Test_Time_Series.iloc[:,1]

# Fit the model for daily data considering cycle of months and years:
TBATS_estimator = TBATS(seasonal_periods=(30,365)) # Month and Year seasonal Components
# Note: this takes quite a bit! for convenience, I fitted the model, then saved (and loaded it)
TBATS_Model = TBATS_estimator.fit(Train_Target)

###############################################################################
# Pickle and unpickle object (some of these estimations can take some time!)
with open('TBATS_Model.pickle', 'wb') as output:
    pickle.dump(TBATS_Model, output)  
with open('TBATS_Model.pickle', 'rb') as data:
    TBATS_Model = pickle.load(data) 
###############################################################################

# Forecast the fitted model for the remaining 367 days, i.e, length of test data:
TBATS_forecast = TBATS_Model.forecast(steps=367)

# # Approach 2: AutoML approach AutoTS (see https://pypi.org/project/AutoTS/)
# Model declaration
model = AutoTS(
    forecast_length=367,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,
    model_list="superfast",
	transformer_list="fast",
    max_generations=25,
    num_validations=2,
    validation_method="backwards"
)

# Declares the series time stamp as index (required by model)
Train_Time_Series_AutoTS = Train_Time_Series.set_index('Date/Time') 
Test_Time_Series_AutoTS = Test_Time_Series.set_index('Date/Time') 

# This snippet takes some time. Fits the model in the Training dataset:
AUTOTS_Model = model.fit(Train_Time_Series_AutoTS)

###############################################################################
# Same pickling as before, to save time in future iterations of code:
with open('AUTOTS_Model.pickle', 'wb') as output:
    pickle.dump(AUTOTS_Model, output)
# Loading the datasets:
with open('AUTOTS_Model.pickle', 'rb') as data:
    AUTOTS_Model = pickle.load(data)   
###############################################################################  
    
# Use the trained model to forecast for the next 367 periods (again, length of test data, obviously)    
Autots_forecast = AUTOTS_Model.predict().forecast

## Approaches 3,4,5: SARIMA with (SARIMAX) and without exogenous regressors: use Automatic approach similar 
## to GridSearch (auto_arima) to find optimal values.
## Consider only monthly seasonal cycles. Save time to get only low order AR and MA terms.

# Model declaration and Grid Search. Takes quite a bit!
Sarima_model_no_exog = auto_arima(Train_Target, start_p=0, start_q=0, 
                         max_p=2, max_q=2, start_P=0, 
            start_Q=0, max_P=2, max_Q=2, m=30, seasonal=True,
            trace=True, d=0, D=0, error_action='warn', 
            suppress_warnings=True, 
            random_state = 8748, n_fits=5)

SarimaX_model = auto_arima(Train_Target, start_p=0, start_q=0, 
                         max_p=2, max_q=2, start_P=0, 
            start_Q=0, max_P=2, max_Q=2, m=30, seasonal=True, 
            exogenous = Train_Time_Series.iloc[:,2:15] ,
            trace=True, d=0, D=0, error_action='warn', 
            suppress_warnings=True, 
            random_state = 8748, n_fits=5)

# Usual pickling and loading:
###############################################################################
with open('SARIMA_no_EXOG.pickle', 'wb') as output:
    pickle.dump(Sarima_model_no_exog, output)
with open('SARIMAX.pickle', 'wb') as output:
    pickle.dump(SarimaX_model, output)
with open('SARIMA_no_EXOG.pickle', 'rb') as data:
    Sarima_model_no_exog = pickle.load(data)   
with open('SARIMAX.pickle', 'rb') as data:
    SarimaX_model = pickle.load(data)
###############################################################################

# Using the 2 SARIMA models to perform predictions for the test set:
Sarima_no_exog_forecast = Sarima_model_no_exog.predict(n_periods=367)
# Considering the covariates from the test set (if we have those):
SarimaX_forecast_test_Covariates = SarimaX_model.predict(n_periods=367,
                                                         exogenous = Test_Time_Series.iloc[:,2:15].fillna(method = 'ffill'))
# SARIMAX Results seem too good to be true (overfitting?)
# Using the model to predict test data using covariates from the train data (obviously with same length):
# Let us use the last year in the train data to predict new test:
SarimaX_forecast_test_train_Covariates = SarimaX_model.predict(n_periods=len(Test_Time_Series),
                                                          exogenous = 
                                                          Train_Time_Series.iloc[-367:,2:15].fillna(method = 'ffill'))
# Note what we have done here. "Sarima_no_exog_forecast" is a model trained without exogenous regressors
# and used to predict the test data. "SarimaX_forecast_test_Covariates" predicts the test data of temperature
# using the exogenous regressors from test data. And "SarimaX_forecast_test_train_Covariates" does the same
# but considering the last 367 observations from train data as exogenous regressors.

# Approach 5: Vector Auto-Regressive (VAR), models all time series together.
# Begin by computing the Granger causality between series, measuring how much one series is
# Helpful in predicting another. 
maxlag=5
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
# Take into account that the complete time series is given by DailyFullExplanatory (lines 189-192)
Granger_test_matrix = grangers_causation_matrix(Daily_Full_explanatory.iloc[:,1:15].dropna(), 
variables = Daily_Full_explanatory.iloc[:,1:15].columns)        
print(Granger_test_matrix.iloc[:1].to_string())
Granger_causality_vector = Granger_test_matrix.iloc[:1].T
# The table above gives the p-values of the Granger test. If smaller than 0.05, 
# then they Granger-cause Temperature.
# We keep only the variables that Granger-cause Prices and check them against previous:
Granger_causes_Temperature = Granger_causality_vector[Granger_causality_vector.iloc[:,0] < 0.05]
Granger_causes_Temperature.rename(columns = {'T (degC)_y':'Granger_Value_p-test'}, inplace = True)
# Note that this matches the correlation and graphical analysis (all variables except
# wv (m/s), max. wv (m/s) and wd (deg) Granger cause Temperature. We could exclude these 3.

# Next step: Cointegration test. Since all series are I(0), this step can be waived, but will leave it here for completeness:
# Cointegration test (Johansen):
from statsmodels.tsa.vector_ar.vecm import coint_johansen
def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)
    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
cointegration_test(Daily_Full_explanatory.iloc[:,1:15].dropna())
# There is no need to difference any pair of series, but I will leave it here

# Fit a VAR on train and test data and derive quality measures:
# Number of Observations and declaration of train and test chunks:
nobs = 367
TimeSeries_train, TimeSeries_test = Train_Time_Series, Test_Time_Series

# Find model order by fitting model for several delays and computing the AIC:
model = VAR(np.asarray(TimeSeries_train.iloc[:,1:15]))
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
# Results aren't that different, let's choose 5 as the VAR order 

# Train VAR with selected model order:
model_fitted = model.fit(5)
model_fitted.summary()

# Checking for serial correlation of residuals using the Durbin-Watson statistic:
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)
for col, val in zip(TimeSeries_train.iloc[:,1:15].columns, out):
    print(col, ':', round(val, 2))
# The residuals distribution looks OK!

# Fitting the model on the train dataset:
lag_order = model_fitted.k_ar
# Input data for forecasting
forecast_input = TimeSeries_train.iloc[:,1:15].values[-lag_order:]

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
VAR_forecast = pd.DataFrame(fc, index=Daily_Full_explanatory.iloc[:,1:15].index[-nobs:], 
                           columns=Daily_Full_explanatory.iloc[:,1:15].columns)
# Sanity check
VAR_forecast
# VAR_forecast has the same name structure as the other methods (method_forecast)
# but returns a list of 14 time series, the temperature and all the predicted others
# in a length of 367 observations. Thus, VAR models and predicts all series together!

# Remove White spaces in predicted, real vectors:
VAR_forecast.columns = VAR_forecast.columns.str.replace(' ', '')
Daily_Full_explanatory.columns = Daily_Full_explanatory.columns.str.replace(' ', '')
TimeSeries_train.columns = TimeSeries_train.columns.str.replace(' ', '')
TimeSeries_test.columns = TimeSeries_test.columns.str.replace(' ', '')

# PLot of forecasts vs actuals for the VAR (All time series)
fig, axes = plt.subplots(nrows=int(len(Daily_Full_explanatory.iloc[:,1:15].columns)/2), 
                         ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(Daily_Full_explanatory.iloc[:,1:15].columns, axes.flatten())):
    VAR_forecast[col].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    TimeSeries_test.iloc[:,1:15][col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
plt.tight_layout();
# VAR cannot model seasonality, finds it safer to just forecast mean of the series
# Seems to be the approach that minimizes the error metric.

# Evaluate Accuracy of the models (only some explanatory variables, the others are easy enough!)
# Uniform names:
model_AUTOTS_prediction = Autots_forecast.iloc[:,0]
dates = list(model_AUTOTS_prediction.index)
plot_test = Test_Time_Series.iloc[:,1]
plot_test.index = dates
model_TBATS_prediction = pd.DataFrame(TBATS_forecast,index=dates)
model_AUTOTS_prediction = Autots_forecast.iloc[:,0]
model_SARIMA_prediction = pd.DataFrame(Sarima_no_exog_forecast,index=dates)
model_SARIMAX_prediction = pd.DataFrame(SarimaX_forecast_test_Covariates,index=dates)
model_SarimaX_forecast_test_train_Covariates = pd.DataFrame(SarimaX_forecast_test_train_Covariates,index=dates)
model_VAR_prediction = VAR_forecast.iloc[:,0]
model_VAR_prediction.index=dates

# Pickle the objects for further study:
temp_list = [plot_test,model_TBATS_prediction,
             model_AUTOTS_prediction,model_SARIMA_prediction,
             model_SARIMAX_prediction,model_SarimaX_forecast_test_train_Covariates,model_VAR_prediction]
results = pd.concat(temp_list,axis=1)
results.columns = ["Observed","TBATS_forecast","AUTOTS_forecast",
                   "SARIMA_forecast","SARIMAX_forecast","SanityCheckSarimax","VAR_forecast"]
# Note: results puts together in a dataframe the observed test data (year of 2016, daily) agains
# the predictions of the methods as explained above. I called "SanityCheckSarimax" the method
# of SarimaX that uses as exogenous regressors the last 367 values of the train data, as explained above. 
# Everything else is self-explanatory.

###############################################################################
with open('results.pickle', 'wb') as output:
    pickle.dump(results, output)    
with open('results.pickle', 'rb') as data:
    results = pickle.load(data) 
###############################################################################
    
# Several measures of adjustment, both classical and time-series oriented
def forecast_accuracy(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    # Treatment of series to use further methods:
    Join = pd.concat([forecast, actual], axis=1).dropna()
    EuclideanDist = np.linalg.norm(Join.iloc[:,0].values-Join.iloc[:,1].values) #Euclidean Distance
    CosineSimilarity = np.dot(Join.iloc[:,0], Join.iloc[:,1])/(np.linalg.norm(Join.iloc[:,0])*np.linalg.norm(Join.iloc[:,1]))
    Pearson = pearsonr(Join.iloc[:,0],Join.iloc[:,1])[0]
    return(mae,rmse,
           EuclideanDist, 
           CosineSimilarity,
           Pearson)
# Ensemble of quality measures to ascertain goodness of fit of each model.

# We had fit each model, now let's see how they match up by evaluating their fit compared:
forecast_results = []
for column in results.iloc[:,1:len(results)].columns:
    forecast_results.append(forecast_accuracy(results[column],results.iloc[:,0]))
# This dataframe contrasts measures and methods:
forecast_results = pd.DataFrame (forecast_results,
                                 columns=['MAE','RMSE','Euclidean','CosineSim.','Pearson'])
forecast_results.index = ['TBATS','AUTOTS','SARIMA','SARIMAX','SARIMAX_2','VAR']
forecast_results.sort_values(by=['MAE','RMSE','Euclidean','CosineSim.','Pearson'])
# The output of this step is a dataframe matching methods and quality measures that says how
# well each method performed. Note we want to minimize MAE, RMSE, Euclidean, and maximize the other 2.
# This points to the conclusions: VAR performed horribly, being the worse, followed by SARIMA;
# The automated ML approaches are mid of the scale. SARIMAX is probably some overfitting phenomenon.
# If one forgets that, then SARIMAX_2 is the winner. This method involves using the last n observations
# of the available data as exogenous regressors for the forecast of the next n periods. 

# Let us plot the predicted values of each method vs test data to get a visual feel:
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.plot(results.iloc[:,1],'r', label='Forecast')
plt.plot(results.iloc[:,0],'b',label='Actual')
plt.xlabel('Day in 2016')
plt.ylabel('Mean Temperature')
plt.legend(loc='best')
plt.title('Predictions of TBATS Model')
plt.subplot(1,2,2)
plt.plot(results.iloc[:,2],'r', label='Forecast')
plt.plot(results.iloc[:,0],'b',label='Actual')
plt.xlabel('Day in 2016')
plt.ylabel('Mean Temperature')
plt.legend(loc='best')
plt.title('Prediction of AUTOTS')
# The 2 automated methods replicate some seasonal cycles, but are not able to match the full
# variation, especially in a monthly cycle. In this regard, TBATS is worse.

# SARIMA vs SARIMAX, the latter probably overfit:
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.plot(results.iloc[:,3],'r', label='Forecast')
plt.plot(results.iloc[:,0],'b',label='Actual')
plt.xlabel('Day in 2016')
plt.ylabel('Mean Temperature')
plt.legend(loc='best')
plt.title('Predictions of SARIMA Model')
plt.subplot(1,2,2)
plt.plot(results.iloc[:,4],'r', label='Forecast')
plt.plot(results.iloc[:,0],'b',label='Actual')
plt.xlabel('Day in 2016')
plt.ylabel('Mean Temperature')
plt.legend(loc='best')
plt.title('Prediction of SARIMAX')
# SARIMA is able to replicate the seasonal effect in the beginning, but then 
# simply predicts the mean value. SARIMAX is practically indistinguishable from the data.

# SARIMAX (Approach 2) vs VAR:
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.plot(results.iloc[:,5],'r', label='Forecast')
plt.plot(results.iloc[:,0],'b',label='Actual')
plt.xlabel('Day in 2016')
plt.ylabel('Mean Temperature')
plt.legend(loc='best')
plt.title('Predictions of SARIMA_SanityCheck Model')
plt.subplot(1,2,2)
plt.plot(results.iloc[:,6],'r', label='Forecast')
plt.plot(results.iloc[:,0],'b',label='Actual')
plt.xlabel('Day in 2016')
plt.ylabel('Mean Temperature')
plt.legend(loc='best')
plt.title('Prediction of VAR')

######################## MODEL CHOICE ##############################################################
# From the above, the logic is clear: we use the SARIMAX model with the last n periods for exogenous variables 
# to predict the following n periods.

# Predict on an unseen time labeled data with chosen model (next 31 days)
# Write SarimaX_model to get model order and declare it:
SarimaX_model
Predictive_model = sm.tsa.statespace.SARIMAX(Daily_Full_explanatory.iloc[:,1],
                                order=(1, 0, 2),
                                seasonal_order=(0, 0, 2, 30),
                                exog = Daily_Full_explanatory.iloc[:,2:15].fillna(method = 'ffill'),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

# Fit the model declared to the entire dataset to perform predictions with:
# Beware: takes a few minutes!
results = Predictive_model.fit()
# This is a results wrapper that can be used to get forecasts with using SARIMAX determined automatically:

# Get forecasts for the unseen, out of sample 31 days by using the last 31 days of data, as outlined above:
pred_uc = results.get_forecast(steps=31,
                               exog=Daily_Full_explanatory.iloc[-31:,2:15].fillna(method = 'ffill'))
Predicted_weather = pred_uc.predicted_mean
datelist = pd.date_range(datetime(2017, 1, 1), periods=31).tolist()
Predicted_weather.index = datelist
# Pedicted_weather matches each day against mean temperature.

# Let us plot this prediction:
fig = plt.figure()
ax = fig.add_subplot(1,1,1)  
plt.plot(Predicted_weather)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

# Let us see how the prediction looks agains the corresponding month of the previous 2 years:
Weather_Jan_2016 = Daily_Full_explanatory[(Daily_Full_explanatory['Date/Time'].astype(str).str[:7] == '2016-01')].iloc[:,1]
Weather_Jan_2015 = Daily_Full_explanatory[(Daily_Full_explanatory['Date/Time'].astype(str).str[:7] == '2015-01')].iloc[:,1]

# Comparison of prediction vs Weather of January in 2016 and 2015:
plt.figure(figsize=(12, 6))
plt.subplot(1,1,1)
plt.plot(np.array(pred_uc.predicted_mean),'r', label='Forecast Jan 2017')
plt.plot(np.array(Weather_Jan_2016),'b',label='Weather Jan2016')
plt.plot(np.array(Weather_Jan_2015),'g',label='Weather Jan2015')
plt.xlabel('Observation')
plt.ylabel('Temperatures over January')
plt.legend(loc='best')
plt.title('Predictions and actual values January')
# The model is able to replicate the initial high temperatures observed
	
