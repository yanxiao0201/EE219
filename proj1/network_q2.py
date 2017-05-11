import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import pandas as pd
from network_common import *

#Question number 2a
def linear_train_test(features, target, n):
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.9, random_state=n)
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    #regr_scores = cross_val_score(regr, other_df, size_df, cv=10)
    
    #Get summary from statsmodel
#    y_train = pd.DataFrame(y_train)
#    X_train = pd.DataFrame(X_train)
#    y_train.columns = ['SIZE']
#    pandas_list = ['WEEK', 'MON', 'TUE', 'WED', 'THUR', 'FRI', 'SAT', 'SUN', 'START']
#    for i in range(5):
#        pandas_list.append('WORK'+str(i))
#    for i in range(30):
#        pandas_list.append('FILE'+str(i))
#    pandas_list.append('TIME')
#    X_train.columns = pandas_list
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    print results.summary()
    #root mean square error
    mse = mean_squared_error(y_test, y_predict)
    rmse = mse**0.5
    print rmse
    
#    date = []
#    for i in range(len(X_test)):
#        day = np.argmax(X_test[i, 1:7])
#        date.append((X_test[i, 0]-1)*7 + day)
    day = X_test['Monday']+X_test['Tuesday']*2+X_test['Wednesday']*3+X_test['Thursday']*4+X_test['Friday']*5+X_test['Saturday']*6+X_test['Sunday']*7
    date = (X_test['WEEK']-1)*7 + day
    plt.figure('round '+str(n)+' values versus time')
    plt.title('round '+str(n)+' fitted and actual values versus time')
    line_test, = plt.plot(date, y_test, 'bx', label='actual')
    line_predict, = plt.plot(date, y_predict, 'yx', label='fitted')
    plt.legend([line_test, line_predict])
    plt.xlabel('days')
    plt.ylabel('size (GB)')
    plt.savefig('round_'+str(n)+'_values versus time')
    
    residual = y_test - y_predict
    plt.figure('round '+str(n)+' residual')
    plt.title('round '+str(n)+' residual versus fitted values')
    plt.plot(y_predict, residual, 'bx')
    plt.xlabel('fitted value (GB)')
    plt.ylabel('residual (GB)')
    plt.savefig('round_'+str(n)+'_residual')

    return mse

def linear_regression_model():
    rounds = 10
    sum_mse = 0
    sum_rmse = 0
    rmse_array = []
    for n in range(rounds):
        mse = linear_train_test(other, size, n)
        rmse = mse**0.5
        rmse_array.append(rmse)
        sum_rmse += rmse
        sum_mse += mse
    mse_mean = sum_mse / rounds
    rmse_mean = sum_rmse / rounds
    rmse_oof = mse_mean**0.5
    rmse_std = np.std(rmse_array)
    print 'RMSE OOF %.9f' % rmse_oof
    print 'RMSE mean %.9f' % rmse_mean
    print 'RMSE std %.9f' %rmse_std
#print np.std(rmse_array)

#plt.show()
