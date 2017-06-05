from matplotlib import pyplot as plt
from common import *
import numpy as np
import json

#--------------------------------

# load data from files...

util.tic(timer_id=1)

scatterplot_hashtags = ['#SuperBowl', '#NFL']
tweets = {}

for fname in flist:
	#fname = retreive_filename_by_hashtag(hashtag)
	hashtag = util.get_hashtags(fname)
	data = {
		"time" : [],
		"follower_count" : [],
		"retweet_count" : [],
		"count" : 0
	}

	# just trying to do something fancy...
	out = util.get_linecount(fname, timeit=True)

	util.tic('Loading tweets w/ %s' % hashtag)
	for line in open(fname, 'r'):
		tweet = json.loads(line)
		data['time'].append(tweet['firstpost_date'])
		data['follower_count'].append(tweet['author']['followers'])
		data['retweet_count'].append(tweet['tweet']['retweet_count'])
		data['count'] += 1

		# print some progress to screen...
		if data['count'] % 1000 == 0 or data['count'] == out:
			print '%d out of %s line(s) finished' % (data['count'], out)
	util.toc()
	
	timespan = (min(data['time']), max(data['time']))
	timelength = timespan[1] - timespan[0] 		# in seconds

	# record data and stats
	tweets[hashtag] = {}
	tweets[hashtag]['data'] = data
	tweets[hashtag]['timespan'] = timespan
	tweets[hashtag]['timelength'] = timelength

print "Time it takes to load data:"
util.toc(timer_id=1)
print ""

#--------------------------------

# analyze and plot data

for fname in flist:
	hashtag = util.get_hashtags(fname)

	# extract data
	data = tweets[hashtag]['data']
	timespan = tweets[hashtag]['timespan']
	timelength = tweets[hashtag]['timelength']

	num_hours = timelength / 3600 + 1
	if hashtag in scatterplot_hashtags:
		# draw histogram
		x = np.array(data['time']) - timespan[0]
		x /= 3600
		plt.hist(x, num_hours, lw=0)
		plt.title(hashtag)
		plt.xlabel('Hour since start')
		plt.ylabel('Number of tweets')

		# show / save plot
		plt.savefig('part1_%s.png' % hashtag)
		plt.show()

	# print statistics
	print "Hashtag: %s" % hashtag
	print "Average tweets per hour: %f" % (data['count'] * 3600.0 / timelength)
	print "Average number of followers: %f" % (np.mean(data['follower_count']))
	print "Average number of retweets: %f" % (np.mean(data['retweet_count']))
	print "Duration: ~%d hours" % num_hours
	print ""
