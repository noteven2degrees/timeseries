# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 19:17:33 2018
@author: noteven2degrees
"""

import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# figure parameters
rcParams['figure.figsize'] = 12, 6

# read data
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('./data/AirPassengers.csv', parse_dates='Month', index_col='Month',date_parser=dateparse)
print data.head()
ts = data['Passengers']

# decompose time series (optional)
decomposition = seasonal_decompose(ts, two_sided=False)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
residual.dropna(inplace=True)

# create training and test set
size = int(len(residual) * 0.8)
train, test = residual[0:size], residual[size:len(residual)]
plt.plot(train.index, train, color='blue', label='Training set')
plt.plot(test.index, test, color='green', label='Testing set')
plt.legend()
plt.title('Ground truth training and testing set')
plt.show()


# assess quality of ARIMA models
# for iterative one-step forecasting on test data set
def compare_ARIMA_modes_testing(order):
    history = [x for x in train]
    predictions_f = list()
    predictions_p = list()
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit(disp=-1)
        yhat_f = model_fit.forecast()[0][0]
        yhat_p = model_fit.predict(start=len(history), end=len(history))[0]
        predictions_f.append(yhat_f)
        predictions_p.append(yhat_p)
        history.append(test[t])
    error_f = mean_squared_error(test, predictions_f)
    error_p = mean_squared_error(test, predictions_p)
    print('MSE forecast:\t\t\t{:1.4f}'.format(error_f))
    print('MSE predict:\t\t\t{:1.4f}'.format(error_p))
    return {'Predictions forecast': predictions_f,
            'Predictions predict': predictions_p,
            'MSE forecast': error_f,
            'MSE predict': error_p}

ar_testing = compare_ARIMA_modes_testing((1, 0, 0))
ma_testing = compare_ARIMA_modes_testing((0, 1, 0))
ig_testing = compare_ARIMA_modes_testing((0, 0, 1))
arma_testing = compare_ARIMA_modes_testing((1, 1, 0))
igma_testing = compare_ARIMA_modes_testing((0, 1, 1))
arig_testing = compare_ARIMA_modes_testing((1, 0, 1))
arima_testing = compare_ARIMA_modes_testing((1, 1, 1))

# forecast and predict are identical for AR
plt.plot(test, label='Ground Truth')
plt.plot(ar_testing['Predictions forecast'], color='red', label='.forecast()')
plt.plot(ar_testing['Predictions predict'], color='green', label='.predict()')
plt.legend()
plt.title('AR')
plt.show()

# forecast and predict are different for MA
plt.plot(test, label='Ground Truth')
plt.plot(arma_testing['Predictions forecast'], color='red', label='.forecast()')
plt.plot(arma_testing['Predictions predict'], color='green', label='.predict()')
plt.legend()
plt.title('ARMA')
plt.show()


# compare forecasting results of ARIMA models
# for iterative forecasting
def compare_ARIMA_modes(order):
    history_f = [x for x in train]
    history_p = [x for x in train]
    predictions_f = list()
    predictions_p = list()
    for t in range(len(test)):
        model_f = ARIMA(history_f, order=order)
        model_p = ARIMA(history_p, order=order)
        model_fit_f = model_f.fit(disp=-1)
        model_fit_p = model_p.fit(disp=-1)
        yhat_f = model_fit_f.forecast()[0][0]
        yhat_p = model_fit_p.predict(start=len(history_p), end=len(history_p))[0]
        predictions_f.append(yhat_f)
        predictions_p.append(yhat_p)
        history_f.append(yhat_f)
        history_f.append(yhat_p)
    error_f = mean_squared_error(test, predictions_f)
    error_p = mean_squared_error(test, predictions_p)
    print('MSE forecast:\t\t\t{:1.4f}'.format(error_f))
    print('MSE predict:\t\t\t{:1.4f}'.format(error_p))
    return {'Predictions forecast': predictions_f,
            'Predictions predict': predictions_p,
            'MSE forecast': error_f,
            'MSE predict': error_p}

ar = compare_ARIMA_modes((1, 0, 0))
ma = compare_ARIMA_modes((0, 1, 0))
ig = compare_ARIMA_modes((0, 0, 1))
arma = compare_ARIMA_modes((1, 1, 0))
igma = compare_ARIMA_modes((0, 1, 1))
arig = compare_ARIMA_modes((1, 0, 1))
arima = compare_ARIMA_modes((1, 1, 1))

# forecast and predict are different for AR
plt.plot(test, label='Ground Truth')
plt.plot(ar['Predictions forecast'], color='red', label='.forecast()')
plt.plot(ar['Predictions predict'], color='green', label='.predict()')
plt.legend()
plt.title('AR')
plt.show()

# forecast and predict are different for ARMA
plt.plot(test, label='Ground Truth')
plt.plot(arma['Predictions forecast'], color='red', label='.forecast()')
plt.plot(arma['Predictions predict'], color='green', label='.predict()')
plt.legend()
plt.title('ARMA')
plt.show()



# compare forecasting results of ARIMA models
# using the step parameter
def compare_ARIMA_modes_steps(order):
    history = [x for x in train]
    model = ARIMA(history, order=order)
    model_fit = model.fit(disp=-1)
    predictions_f_ms = model_fit.forecast(steps=len(test))[0]
    predictions_p_ms = model_fit.predict(start=len(history), end=len(history)+len(test)-1)
    error_f_ms = mean_squared_error(test, predictions_f_ms)
    error_p_ms = mean_squared_error(test, predictions_p_ms)
    print('MSE forecast:\t\t\t{:1.4f}'.format(error_f_ms))
    print('MSE predict:\t\t\t{:1.4f}'.format(error_p_ms))
    return {'Predictions forecast': predictions_f_ms,
            'Predictions predict': predictions_p_ms,
            'MSE forecast': error_f_ms,
            'MSE predict': error_p_ms}

ar_steps = compare_ARIMA_modes_steps((1, 0, 0))
ma_steps = compare_ARIMA_modes_steps((0, 1, 0))
ig_steps = compare_ARIMA_modes_steps((0, 0, 1))
arma_steps = compare_ARIMA_modes_steps((1, 1, 0))
igma_steps = compare_ARIMA_modes_steps((0, 1, 1))
arig_steps = compare_ARIMA_modes_steps((1, 0, 1))
arima_steps = compare_ARIMA_modes_steps((1, 1, 1))

# forecast and predict are identical for AR
plt.plot(test, label='Ground Truth')
plt.plot(ar_steps['Predictions forecast'], color='red', label='.forecast(steps)')
plt.plot(ar_steps['Predictions predict'], color='green', label='.predict(steps)')
plt.legend()
plt.title('AR')
plt.show()

# forecast and predict are different for ARMA
plt.plot(test, label='Ground Truth')
plt.plot(arma_steps['Predictions forecast'],  color='red', label='.forecast(steps)')
plt.plot(arma_steps['Predictions predict'], color='green', label='.predict(steps)')
plt.legend()
plt.title('ARMA')
plt.show()
