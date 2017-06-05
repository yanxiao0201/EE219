from sklearn import linear_model
from matplotlib import pyplot as plt
from common import *
import statsmodels.api as sm
import numpy as np
import time, json, sys

interactive_mode = False
interactive_mode_on = False
n_hour_window = 1
save_feature_data = False
save_model_stats = False

question = '3'
final_report = {}
predictions = {}

#--------------------------------

######################
## Feature Handling ##
######################

# defines how to initialize a feature vector
def initialize_feature1():
	"""
	part 2 features:
		1. num of tweets
		2. total num of retweets
		3. sum of followers
		4. max of followers
		5-28.time of day
	"""
	return [0,0,0,0] + [0] * 24

def initialize_feature2():
	"""
	part 3 features:
		1. num of tweets
		2. total num of distinct authors
		3. total num of retweets
		4. mention count
		5. url count
	"""
	return [0,[],0,0,0]

def initialize_feature():
	if question == '3':
		return initialize_feature2()
	else:
		return initialize_feature1()

# defines how to use new tweet to update old feature and returns it
def update_feature1(feature, tweet):
	# part 2 update

	# tweet count
	feature[0] += 1
	
	# retweet count
	feature[1] += tweet['tweet']['retweet_count']

	# follower count: sum
	num_follower = tweet['author']['followers']
	feature[2] += num_follower

	# follower count: max
	feature[3] = max(feature[3], num_follower)

	# time of the day
	time_of_day = time.localtime(tweet['firstpost_date']).tm_hour
	feature[time_of_day + 4] = 1

	return feature

def update_feature2(feature, tweet):
	# part 3 update

	# tweet count
	feature[0] += 1
	
	# unique author count
	user = tweet['author']['nick'] 	# should be unique
	if not user in feature[1]:
		feature[1].append(user)

	# retweet count
	feature[2] += tweet['tweet']['retweet_count']

	# mention count
	feature[3] += len(tweet['tweet']['entities']['user_mentions'])

	# url count
	if len(tweet['tweet']['entities']['urls']) > 0:
		feature[4] += 1

	return feature

def update_feature(feature, tweet):
	if question == '3':
		return update_feature2(feature, tweet)
	else:
		return update_feature1(feature, tweet)

# defines how to intergrate multiple feature vectors into one
def integrate_windows1(features):
	feature = initialize_feature()
	for feat in features:
		feature[0] += feat[0]
		feature[1] += feat[1]
		feature[2] += feat[2]
		feature[3] = max(feature[4], feat[4])
		for idx in xrange(4,len(feat)):
			if feat[idx] == 1:
				feature[idx] = 1
	return feature

def integrate_windows2(features):
	feature = initialize_feature()
	for feat in features:
		feature[0] += feat[0]
		
		for nick in feat[1]:
			if not nick in feature[1]:
				feature[1].append(nick)

		feature[2] += feat[2]
		feature[3] += feat[3]
		feature[4] += feat[4]
		#feature[4] += feat[4] * feat[0]
	#feature[4] = float(feature[4]) / feature[0]
	return feature

def integrate_windows(features):
	if question == '3':
		return integrate_windows2(features)
	else:
		return integrate_windows1(features)

def postprocess_feature(feature):
	if question == '3':
		# num of unique authors
		feature[1] = len(feature[1])
		# url ratio
		#feature[4] = float(feature[4]) / feature[0]

	return feature

#--------------------------------

######################
## Machine Learning ##
######################

# Load data from file
def load_tweet_data(fname):
	hashtag = util.get_hashtags(fname)
	features = {}
	targets = {} 	# tweet count in this project
	other = {}

	# other data
	timespan = [float('inf'), float('-inf')]

	out = util.get_linecount(fname, timeit=True)

	# read from twitter data file
	util.tic('Loading tweets w/ %s' % hashtag)
	count = 0
	for line in open(fname, 'r'):
		tweet = json.loads(line)
		count += 1

		hour = tweet['firstpost_date'] / 3600 	# y

		if hour not in features:
			features[hour] = initialize_feature()
			targets[hour] = 0

		feature = update_feature(features[hour], tweet)
		features[hour] = feature
		targets[hour] += 1 		# count number of tweets

		# update time span
		if hour < timespan[0]:
			timespan[0] = hour
		elif hour > timespan[1]:
			timespan[1] = hour

		# print some progress to screen...
		if count % 500 == 0 or count == out:
			print '%d out of %s line(s) finished' % (count, out)
	util.toc()

	other['timespan'] = tuple(timespan)

	return hashtag, features, targets, other

# Use linear regression to predict
def do_linreg_pred(hashtag, features, targets, other):
	util.tic('%s : processing features...' % hashtag)

	X, y, keys = [], [], []
	for key in features:
		X.append(features[key])
		y.append(targets[key])
		keys.append(key)

	if n_hour_window > 1:
		# only do this when n_hour_window is not 1
		# in order to avoid unnecessary slow down
		X = [integrate_windows(X[i:i + 
			n_hour_window]) for i in xrange(0,len(X) - n_hour_window + 1)]
	for idx, val in enumerate(X):
		X[idx] = postprocess_feature(val)

	# trim to distinguish training and prediciton data
	X_pred = X[-1]
	X = X[:-1]
	y = y[n_hour_window:]
	keys = keys[n_hour_window:]

	# save postprocessed features and targets if specified to do so
	if save_feature_data:
		save_data(hashtag, X, y, keys, other)

	util.toc()

	# linear regression model
	lr = linear_model.LinearRegression()
	lr.fit(X, y)
	y_pred_raw = lr.predict([X_pred])
	y_pred = max(y_pred_raw, 0)

	# generate report
	final_report[hashtag] = ""
	final_report[hashtag] += "---- %s ----\n" % hashtag
	final_report[hashtag] += do_model_statistics(X, y, hashtag)
	final_report[hashtag] += ("\nThe model predicts that the number of tweets with " + 
		"hashtag %s will be %d in next hour\n") % (hashtag, y_pred)

	predictions[hashtag] = y_pred

	if question == '3':
		# scatter plot (scat plot LOL)
		feature_names = ['tweet count', 'author count', 'retweet count',
						 'mention count', 'url count']
		for idx, name in enumerate(feature_names):
			x = [feat[idx] for feat in X]
			plt.scatter(x, y, lw=0, s=9)
			plt.xlabel(name)
			plt.ylabel('Next hour tweet count')
			plt.title(hashtag)
			plt.savefig('scatter_%s_%d.png' % (hashtag, idx))
			plt.clf()

def save_data(hashtag, features, targets, keys, other):
	fd1 = open('features_%s.csv' % hashtag, 'w')
	for feat in features:
		feat_str = '%s' % feat
		fd1.write('%s\n' % feat_str)
	fd1.close()

	fd2 = open('targets_%s.csv' % hashtag, 'w')
	t_arr = []
	for item in targets:
		t_arr.append(item)
	t_arr = '%s' % t_arr
	fd2.write('%s\n' % t_arr)
	fd2.write('%s\n' % keys)
	fd2.close()

def do_model_statistics(X, y, hashtag):
	model = sm.OLS(y, X)
	results = model.fit()
	report = ""
	report += "n-hour-window: %d\n" % n_hour_window
	report += "R-squared: %f\n" % results.rsquared
	report += "Adjusted R-squared: %f\n" % results.rsquared_adj
	report += "p_values:\n"
	report += "%s\n" % results.pvalues
	report += "t_values:\n"
	report += "%s\n" % results.tvalues

	if save_model_stats:
		fd = open('model_stats%d_%s.txt' % (n_hour_window, hashtag), 'w')
		fd.write(report)
		fd.close()

	return report

#--------------------------------

###############
## Dev Tools ##
###############

def interact_main():
	global interactive_mode, n_hour_window

	while True:
		cmd = raw_input("Enter a number to change window size, 'q' to quit: ")
		if cmd.isdigit():
			cmd = int(cmd)
			if 0 < cmd <= 48:
				n_hour_window = cmd
				break
			else:
				print "Sorry, we have limit on window size."
		elif cmd[0].lower() == 'q':
			interactive_mode = False
			break
		else:
			print "Sorry, didn't recognize your input."

#--------------------------------

if __name__ == '__main__':
	# argument reading
	if (len(sys.argv) > 1):
		for argv in sys.argv[1:]:
			argv = argv.split('=')
			opt = argv[0]
			val = None if len(argv) <= 1 else argv[1]

			if opt == '--window':
				n_hour_window = max(int(val), 1)
			elif opt == '-i':
				interactive_mode_on = True
			elif opt == '--savefeat':
				save_feature_data = True
			elif opt == '--savestats':
				save_model_stats = True
			elif opt == '--question':
				question = str(val)
	
	for fname in flist:
		interactive_mode = interactive_mode_on

		hashtag, features, targets, other = load_tweet_data(fname)
		do_linreg_pred(hashtag, features, targets, other)
		while interactive_mode:
			interact_main()
			do_linreg_pred(hashtag, features, targets, other)

		if interactive_mode_on:
			print "Interactive mode will only do one file..."
			break

	for hashtag in final_report:
		print final_report[hashtag]

	print "Summary - predictions for number of tweets next hour:"
	for hashtag in predictions:
		print "%s : %d" % (hashtag, predictions[hashtag])
