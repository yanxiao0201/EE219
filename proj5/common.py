import util
import os

BASE_DIR = 'tweet_data/'

flist = os.listdir(BASE_DIR)
flist = [BASE_DIR + f for f in flist]
#flist = ['tweet_data/tweets_#gopatriots.txt'] # for experiment

# selectively load tweets by giving a list of desired hashtags
# if not specified or 'all' is passed, load all
# hashtags are case insensitive, and include the '#' symbol
def load_tweets(hashtags=None):
	global flist
	
	sublist = []
	if hashtags == None or hashtags == 'all':
		sublist = flist
	else:
		for hashtag in hashtags:
			hashtag = hashtag.lower()
			for fname in flist:
				if hashtag == util.get_hashtags(fname).lower():
					sublist.append(fname)

	return util.get_tweets(sublist)

# return file name with the given hashtag, None if not found
def retreive_filename_by_hashtag(hashtag):
	hashtag = hashtag.lower()
	for fname in flist:
		if hashtag == util.get_hashtags(fname).lower():
			return fname
	return None
