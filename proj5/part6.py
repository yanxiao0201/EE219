from common import *
import json
import util

def from_washington(loc):
	return loc == 'Washington' or ', Washington' in loc or loc == 'WA' or ', WA' in loc

def from_massachusetts(loc):
	return loc == 'Massachusetts' or ', Massachusetts' in loc or loc == 'MA' or ', MA' in loc

# extracting tweet data
hashtag = '#superbowl'
fname = retreive_filename_by_hashtag(hashtag)

#out = util.get_linecount(fname, timeit=True)

util.tic('Loading tweets w/ %s' % hashtag)

tweet_text = []
tweet_addr = []
i = 0

filestream = open(fname, 'r')
for line in filestream:
	tweet = json.loads(line)
	tweet_content = tweet['title']

	user_location = tweet['tweet']['user']['location']
	if from_washington(user_location):
	    tweet_text.append(tweet_content)
	    tweet_addr.append(1)

	elif from_massachusetts(user_location):
	    tweet_text.append(tweet_content)
	    tweet_addr.append(-1)
		#print tweet_content
        i += 1
        if i%10000 == 0:
            print "Loading tweet #{}".format(i)

filestream.close()

util.toc()
