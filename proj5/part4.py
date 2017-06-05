from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn import svm
from common import *
import matplotlib.pyplot as plt
import numpy as np
import random, datetime
import util

def read_feature_data(hashtag):
	X, y, timestamps = [], [], []

	# Note: eval() is extremely dangerous, just using for convenience
	# of this project only...

	for line in open('features_%s.csv' % hashtag, 'r'):
		X.append(eval(line.strip()))

	i = 1
	for line in open('targets_%s.csv' % hashtag, 'r'):
		if i == 1:
			# first line is targets
			y = eval(line.strip())
		elif i == 2:
			# second line is keys (timestamps)
			timestamps = eval(line.strip())
			break
		i += 1

	if len(X) != len(y) or len(y) != len(timestamps) or len(X) != len(timestamps):
		raise Exception("X, y, timestamps don't agree on dimensions!")

	return X, y, timestamps

def split_by_period(X, y, timestamps):
	t_start = int(datetime.datetime(2015,2,1,8).strftime('%s')) / 3600
	t_end = int(datetime.datetime(2015,2,1,20).strftime('%s')) / 3600

	X_piecewise = {'before' : [], 'between' : [], 'after' : []}
	y_piecewise = {'before' : [], 'between' : [], 'after' : []}
	for idx, t in enumerate(timestamps):
		if t < t_start:
			X_piecewise['before'].append(X[idx])
			y_piecewise['before'].append(y[idx])
		elif t > t_end:
			X_piecewise['after'].append(X[idx])
			y_piecewise['after'].append(y[idx])
		else:
			X_piecewise['between'].append(X[idx])
			y_piecewise['between'].append(y[idx])

	return X_piecewise, y_piecewise

def get_xrosval_err(X, y, random_state=42, model=None):
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=random_state)
	if model == None:
		model = linear_model.LinearRegression()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	y_true = np.array(y_test)

	return np.mean(abs(y_pred - y_true))

def get_10xros_avg(X, y, model=None):
	errors = []
	for i in xrange(10):
		errors.append(get_xrosval_err(X, y, i, model))
	return np.mean(errors)

def main():
	#########################
	## 10-cross validation ##
	#########################

	for hashtag in util.get_hashtags(flist):
		X, y, timestamps = read_feature_data(hashtag)
		err = get_10xros_avg(X, y)
		print "Average prediction error for %s = %f" % (hashtag, err)

	print "\nAfter using piecewise regression model..."

	#######################################
	## piecewise linear regression model ##
	#######################################

	for hashtag in util.get_hashtags(flist):
		X, y, timestamps = read_feature_data(hashtag)
		X_piecewise, y_piecewise = split_by_period(X, y, timestamps)

		print "Average prediction error for regression model for %s:" % hashtag
		for key in ['before', 'between', 'after']:
			model = svm.SVR('rbf') if key == 'between' else None
			#model = None
			err = get_10xros_avg(X_piecewise[key], y_piecewise[key], model)
			print "%-8s %f" % (key, err)

if __name__ == '__main__':
	main()
