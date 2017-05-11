import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import pandas as pd
from housing_common import *
from sklearn.pipeline import Pipeline

def linear_train_test(features, target, n):
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.9, random_state=n)
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)

    model = sm.OLS(y_train, X_train)
    results = model.fit()
    print results.summary()
    #root mean square error
    mse = mean_squared_error(y_test, y_predict)
    rmse = mse**0.5
    print rmse
    
    plt.figure('round '+str(n)+' values versus LSTAT')
    plt.title('round '+str(n)+' fitted and actual values versus LSTAT')
    line_test, = plt.plot(X_test['LSTAT'], y_test, 'bx', label='actual')
    line_predict, = plt.plot(X_test['LSTAT'], y_predict, 'yx', label='fitted')
    plt.legend([line_test, line_predict])
    plt.xlabel('LSTAT')
    plt.ylabel('MEDV')
    plt.savefig('round_'+str(n)+'_values versus LSTAT')
    
    residual = y_test - y_predict
    plt.figure('round '+str(n)+' residual')
    plt.title('round '+str(n)+' residual versus fitted values')
    plt.plot(y_predict, residual, 'bx')
    plt.xlabel('MEDV')
    plt.ylabel('Residual')
    plt.savefig('round_'+str(n)+'_residual')

    return mse

#Question number 4a
def linear_regression_model():
    rounds = 10
    sum_mse = 0
    sum_rmse = 0
    rmse_array = []
    for n in range(rounds):
        mse = linear_train_test(features, target, n)
        rmse = mse**0.5
        sum_rmse += rmse
        rmse_array.append(rmse)
        sum_mse += mse
    mean_mse = sum_mse / rounds
    mean_rmse = sum_rmse / rounds
    rmse_std = np.std(rmse_array)
    rmse_oof = mean_mse**0.5
    print 'RMSE OOF %.9f' % rmse_oof
    print 'RMSE mean %.9f' % mean_rmse
    print 'RMSE std %.9f' %rmse_std
#print np.std(rmses)

#plt.show()

def poly_train_test(features, target, n, max_degree):
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.9, random_state=n)
    
    num_degrees = range(2, max_degree+1)
    rmse_array = []
    for degree in num_degrees:
        #transform to polynomial form
        polynomial_features = preprocessing.PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = linear_model.LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
        pipeline.fit(X_train, y_train)
        y_predict = pipeline.predict(X_test)
        #print degree
        
        #root mean square error
        mse = mean_squared_error(y_test, y_predict)
        rmse = mse**0.5
        rmse_array.append(rmse)
    return rmse_array

#Question 4b
def poly_regression_model():
    #features_poly = features.drop(['INDUS', 'NOX', 'AGE'], axis=1)
    #print features_poly
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.9, random_state=0)
    
    max_degree = 6
    num_degrees = range(2, max_degree+1)
    rmse_array = []
    for degree in num_degrees:
        #transform to polynomial form
        polynomial_features = preprocessing.PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = linear_model.LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
        pipeline.fit(X_train, y_train)
        y_predict = pipeline.predict(X_test)
        #print degree
        
        #root mean square error
        rmse = mean_squared_error(y_test, y_predict)**0.5
        print 'Degree=%d RMSE=%f' % (degree, rmse)
        rmse_array.append(rmse)
    plt.figure('Polynomial Regression for a fixed set')
    plt.title('RMSE versus Degree for a fixed set')
    plt.plot(num_degrees, rmse_array, '-')
    plt.xlabel('Degree of polynomial')
    plt.ylabel('RMSE')
    
    rounds = 10
    sum_rmse = np.zeros(max_degree-1)
    
    for n in range(rounds):
        rmse = np.array(poly_train_test(features, target, n, max_degree))
        sum_rmse += rmse
    mean_rmse = sum_rmse / rounds
    for n in range(len(mean_rmse)):
        print 'Degree=%d Average RMSE=%f' % (n+2, mean_rmse[n])
#rmse_oof = mean_mse**0.5

    plt.figure('Polynomial Regression 10 fold')
    plt.title('Avg RMSE versus Degree for 10 fold')
    plt.plot(num_degrees, mean_rmse, '-')
    plt.xlabel('Degree of polynomial')
    plt.ylabel('Avg RMSE')
    plt.show()

