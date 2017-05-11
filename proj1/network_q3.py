import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import pandas as pd
from sklearn.pipeline import Pipeline
from network_common import *
import time

# returns rmse and r2 value (though r2 now not used...)
def lin_train_test(features, target, n):
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.9, random_state=n)
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)

    rmse = mean_squared_error(y_test, y_predict)**0.5
    score = lr.score(X_test, y_test)

    return rmse, score

def piecewise_regression_model():
    #Question number 3a: 
    other_workflows = []
    size_workflows = []
    rmse_workflows = []
    score_worklows = []
    work_flow_labels = ['work_flow_0', 'work_flow_1', 'work_flow_2', 'work_flow_3', 'work_flow_4']
    for work_flow_label in work_flow_labels:
        other_workflow = other.loc[other[work_flow_label] == 1]
        size_workflow = df.loc[df["WRKF"] == work_flow_label]['SIZE']
        
        other_workflows.append(other_workflow)
        size_workflows.append(size_workflow)
        
        rmse, score = lin_train_test(other_workflow, size_workflow, 0)
        rmse_workflows.append(rmse)
        score_worklows.append(score)

    for idx, rmse in enumerate(rmse_workflows):
        print "%s: RMSE = %f R2 = %f" % (work_flow_labels[idx], rmse, score_worklows[idx])

    
    #Question 3b
    max_degree = 6
    num_degrees = range(2, max_degree+1)
    plot_data = []

    for idx, work_flow_label in enumerate(work_flow_labels):
        other_chosen = other_workflows[idx]
        size_chosen = size_workflows[idx]

        # drop unrelated stuff
        for num in range(1, 30):
            filename = "File_%d" % num
            if filename in other_chosen.columns:
                #other_chosen.drop(filename, 1)
                del other_chosen[filename]

        #print other_chosen
        # reduced to 16 columns

        X_train, X_test, y_train, y_test = train_test_split(other_chosen,
            size_chosen, train_size=0.9, random_state=0)

        rmse_array = []
        #score_array = []
        overall_start_time = time.time()
        for degree in num_degrees:
            print "doing degree %d" % degree
            start_time = time.time()

            polynomial_features = preprocessing.PolynomialFeatures(degree=degree, include_bias=False)
            linear_regression = linear_model.LinearRegression()
            pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
            pipeline.fit(X_train, y_train)
            y_predict = pipeline.predict(X_test)
            
            #root mean square error
            rmse = mean_squared_error(y_test, y_predict)**0.5
            rmse_array.append(rmse)

            #score = pipeline.score(X_test, y_test)
            #score_array.append(score)

            print "time elapsed: %fs" % (time.time() - start_time)
        print "total time elapsed: %fs" % (time.time() - overall_start_time)

        for idx, rmse in enumerate(rmse_array):
            print "%s: degree = %d RMSE = %f" % (work_flow_label, idx+2, rmse)

        plot_data.append(rmse_array)

    lineclr='rgbcm'
    for idx, rmse_array in enumerate(plot_data):
        plt.clf()
        plt.figure('Polynomial Regression')
        plt.xlabel('Degree of polynomial')
        plt.ylabel('RMSE')
        plt.title('Workflow %d' % idx)
        if idx == 0:
            plt.ylim(0.005, 0.025)
        elif idx == 1:
            plt.ylim(0.001, 0.006)
        plt.plot(num_degrees, rmse_array, label=work_flow_label, color=lineclr[idx])
        plt.savefig('poly_regr_rmse_wrkf%d.png' % idx)
    

    # Evaluation

    best_degrees = [4, 4, 5, 5, 6]
    rounds = 5
    idx = 3     # use workflow 3 to experiment

    other_chosen = other_workflows[idx]
    size_chosen = size_workflows[idx]

    # drop unrelated stuff
    for num in range(1, 30):
        filename = "File_%d" % num
        if filename in other_chosen.columns:
            #other_chosen.drop(filename, 1)
            del other_chosen[filename]

    mse_array = []
    degree = best_degrees[idx]

    start_time = time.time()
    for rd in range(rounds):
        X_train, X_test, y_train, y_test = train_test_split(other_chosen,
            size_chosen, train_size=0.9, random_state=rd)

        polynomial_features = preprocessing.PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = linear_model.LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
        pipeline.fit(X_train, y_train)
        y_predict = pipeline.predict(X_test)
        
        #root mean square error
        mse = mean_squared_error(y_test, y_predict)
        mse_array.append(mse)

    rmse_array = [mse ** 0.5 for mse in mse_array]

    oof_rmse = (np.sum(mse_array) / len(mse_array)) ** 0.5 # out-of-fold
    rmse_mean = np.mean(rmse_array)
    rmse_std = np.std(rmse_array)
    print "%s:\nrmse = %f +/- %f\noof_rmse = %f\n" % (work_flow_labels[idx], rmse_mean, rmse_std, oof_rmse)
