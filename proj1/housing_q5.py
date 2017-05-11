import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import pandas as pd
from housing_common import *

def _set_plot(label_y):
	plt.clf() # clear graph
	plt.xlabel("alpha")
	plt.ylabel(label_y)
	plt.xscale("log")
	plt.gca().invert_xaxis()

def a_certain_scientific_regression(regr_func, eval_score=False):
	X_train, X_test, y_train, y_test = train_test_split(features,
		target, train_size=0.9, random_state=0)
	alpha_range = [1, 0.1,0.01,0.001]

	rmse_array = []
	score_array = []
	for a in alpha_range:
		regr = regr_func(alpha = a)
		regr.fit(X_train, y_train)
		y_predict = regr.predict(X_test)
		rmse = mean_squared_error(y_test, y_predict)**0.5
		rmse_array.append(rmse)
		if eval_score:
			score = regr.score(X_test, y_test)
			score_array.append(score)
		print "alpha = %f, rmse = %f" % (a, rmse)
		print "coeff = %s\n" % str([ float("{:0.5f}".format(x)) for x in regr.coef_ ])
	_set_plot("RMSE of prediction")
	plt.plot(alpha_range, rmse_array)
	plt.savefig("housing_regularization_%s.png" % regr_func.__name__)

	if eval_score:
		_set_plot("R2 value")
		plt.plot(alpha_range, score_array, color='g')
		plt.savefig("housing_regularization_R2_plot_%s.png" % regr_func.__name__)

# Question 5a
def ridge_regression():
	a_certain_scientific_regression(linear_model.Ridge, True)

# Question 5b
def lasso_regression():
	a_certain_scientific_regression(linear_model.Lasso, True)
