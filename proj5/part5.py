from part2n3 import load_tweet_data, postprocess_feature
from part4 import *
from common import *
import util

base_dir5 = 'test_data/'
flist5 = os.listdir(base_dir5)
flist5 = [base_dir5 + f for f in flist5]

def main():
	# load models from saved feature data
	models = {}
	for hashtag in util.get_hashtags(flist):
		X, y, timestamps = read_feature_data(hashtag)
		X_piecewise, y_piecewise = split_by_period(X, y, timestamps)

		model_group = {}
		for key in X_piecewise:
			if key == 'between':
				model_group[key] = svm.SVR('rbf')
			else:
				model_group[key] = linear_model.LinearRegression()
			model_group[key] = model_group[key].fit(X_piecewise[key], y_piecewise[key])
		models[hashtag] = model_group

	# make prediction
	predictions = {}
	for fname in flist5:
		hashtag, features, targets, other = load_tweet_data(fname)
		hashtag = '#superbowl'
		model = None
		if 'period1' in fname:
			model = models[hashtag]['before']
		elif 'period2' in fname:
			model = models[hashtag]['between']
		elif 'period3' in fname:
			model = models[hashtag]['after']

		X_pred = None
		for key in features:
			# only need last hour's data because our model takes 1-hour window
			if key == other['timespan'][1]:
				X_pred = postprocess_feature(features[key])
				break

		y_pred_raw = model.predict([X_pred])
		y_pred = max(y_pred_raw, 0)
		predictions[fname] = y_pred
	
	for fname in predictions:
		print "For sample %s our model predicts %d tweets next hour." % (fname, predictions[fname])

if __name__ == '__main__':
	main()
