import util
import os

BASE_DIR = 'test_data/'

flist = os.listdir(BASE_DIR)
flist = [BASE_DIR + f for f in flist]

util.tic('Loading test data...')
test_tweets = util.get_tweets(flist[0])
util.toc()

"""
In [3]: tweet = test_tweets[0]

In [6]: tweet['tweet']['entities']['hashtags']
Out[6]: []
"""
