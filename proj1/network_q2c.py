import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, neural_network
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import pandas as pd
from network_common import *

#Question number 2
def neural_network_regression_model():
    colors = 'rgbmck'
    # change # of nodes and depth
    max_depth = 5
    rounds = 10
    depths = range(2, max_depth + 1)
    nums_nodes = range(100, 601, 50)

    lines = []
    for depth in range(2, max_depth + 1):
        rmse_prediction_array = []
        for num_nodes in nums_nodes:
            rmse_array = []
            for rd in range(rounds):
                X_train, X_test, y_train, y_test = train_test_split(other, size, train_size = 0.9, random_state=rd)
                hidden_layer_sizes = tuple([int(n) for n in np.ones(depth - 1) * num_nodes])
                regr = neural_network.MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)
                regr.fit(X_train, y_train)
                y_predict = regr.predict(X_test)

                # root mean square
                rmse = mean_squared_error(y_test, y_predict)**0.5
                rmse_array.append(rmse)
            
            rmse_mean = np.mean(rmse_array)
            rmse_std = np.std(rmse_array)
            rmse_prediction_array.append(rmse_mean)
            print "depth = %d, num_nodes = %d, RMSE = %f +/- %f" % (depth, num_nodes, rmse_mean, rmse_std)
        plt.xlabel('Number of nodes per layer')
        plt.ylabel('RMSE of prediction')
        line, = plt.plot(nums_nodes, rmse_prediction_array, color=colors[depth - 2], label='depth=%d' % depth)
        lines.append(line)
    plt.legend(lines)
    #plt.show()
    plt.savefig('NN_param_plt.png')
    #plt.clf()
