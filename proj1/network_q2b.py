import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from network_common import *

#Question number 2b
def random_forest_model():
    '''
    est = [20, 100, 200]
    depth = [8, 12, 16, 20]
    feature = [30, 35, 45]
    X_train, X_test, y_train, y_test = train_test_split(other, size, train_size=0.9, random_state=0)
    rmsearray = []
    for i in feature:
        rf = RandomForestRegressor(n_estimators= 20, max_depth= 12, max_features= i)
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict)**0.5
        rmsearray.append(rmse)
    plt.xlabel('feature')
    plt.ylabel('RMSE of prediction')
    plt.plot(feature, rmsearray)
    plt.show()
    
    '''
    est = [i for i in range(20, 200, 40)]
    depth = [j for j in range(4, 20, 4)]
    feature = [k for k in range(20, 46, 5)]
    result = {}
    result['n_est'] = -1
    result['max_depth'] = -1
    result['max_feature'] = -1
    result['rmse'] = 1


    X_train, X_test, y_train, y_test = train_test_split(other, size, train_size=0.9, random_state=0)
    
    for i in est:
        for j in depth:
            for k in feature:
                rf = RandomForestRegressor(n_estimators= i, max_depth= j, max_features= k)
                rf.fit(X_train, y_train)

                y_predict = rf.predict(X_test)
        #root mean square error
                rmse = mean_squared_error(y_test, y_predict)**0.5

                if rmse < result['rmse']:
                    result['rmse'] = rmse
                    result['n_est'] = i
                    result['max_depth'] = j
                    result['max_feature'] = k

    print result

    rmse_list = []
    for estimator in est:
        rf = RandomForestRegressor(n_estimators= estimator, max_depth= result['max_depth'], max_features= result['max_feature'])
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict)**0.5
        rmse_list.append(rmse)
        if estimator == result['n_est']:
            importances = rf.feature_importances_

    plt.figure("tree vs rmse")
    plt.plot(est,rmse_list)
    plt.ylabel("rmse")
    plt.xlabel("number of trees")
    plt.savefig("tree vs rmse")

    rmse_list = []
    for dep in depth:
        rf = RandomForestRegressor(n_estimators= result['n_est'], max_depth= dep, max_features= result['max_feature'])
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict)**0.5
        rmse_list.append(rmse)


    plt.figure("depth vs rmse")
    plt.plot(depth,rmse_list)
    plt.ylabel("rmse")
    plt.xlabel("max depth")
    plt.savefig("depth vs rmse")


    rmse_list = []
    for fea in feature:
        rf = RandomForestRegressor(n_estimators= result['n_est'], max_depth= result['max_depth'], max_features= fea)
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)
        rmse = mean_squared_error(y_test, y_predict)**0.5
        rmse_list.append(rmse)


    plt.figure("feature vs rmse")
    plt.plot(feature,rmse_list)
    plt.ylabel("rmse")
    plt.xlabel("max feature")
    plt.savefig("feature vs rmse")
    
    
    x_axis = range(len(other.columns))
    #print x_axis
    column = other.columns.tolist()
    #print column, type(column)
    plt.figure("import features")
    plt.bar(x_axis,importances)
    plt.xticks(x_axis,column,rotation=70,fontsize = 8)

    plt.savefig("important features")

