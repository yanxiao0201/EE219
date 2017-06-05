from subprocess import Popen, PIPE
import json, time

#--------------------------------

# twitter data processing

# accpets a string representation of a file name or a list of file names
# if given a string of file, returns a list of json objects of tweets
# if given a list of files, returns a dictionary of:
# 	hashtag -> list of json objects of tweets
# returns None for incompatible types
def get_tweets(f):
	if type(f) == str:
		print 'Reading tweet data from: %s' % f
		tweets = []
		for line in open(f, 'r'):
			tweet = json.loads(line)
			tweets.append(tweet)
		return tweets
	elif type(f) == list:
		tweets = {}
		for fname in f:
			hashtag = get_hashtags(fname)
			if type(hashtag) != str or len(hashtag) == 0:
				print ('Warning: cannot extract hastag' + 
				' from file name: %s --> skipping') % fname
				continue
			tweets[hashtag] = get_tweets(fname)
		return tweets
	else:
		return None

# accepts a string of a file name or a list of file names
# given the string of the file name, extract hastag from it
# whether it's a path or just file name doesn't matter
# if something goes wrong, returns an empty string
# if given a list, returns a list of hashtags from every file name
# returns None for incompatible types
def get_hashtags(f):
	if type(f) == str:
		start = f.index('#') if '#' in f else -1
		end = f.index('.txt') if '.txt' in f else -1
		if 0 <= start < end:
			return f[start:end]
		else:
			return ''
	elif type(f) == list:
		hashtags = []
		for fname in f:
			hashtag = get_hashtags(fname)
			if hashtag == None or len(hashtag) == 0:
				print ('Warning: cannot extract hastag' + 
				' from file name: %s --> skipping') % fname
				continue
			hashtags.append(hashtag)
		return hashtags

#--------------------------------

# timing tools

def _print_time_in_miliseconds(start, end):
	if (type(start) != float and type(end) != float):
		print "Error when printing time: given start and end time must be float!"
		return
	duration = end - start
	addtional_info = " (%fs)" % duration if duration >= 10 else ""
	print "Time elapsed: %fms%s" % (duration * 1000, addtional_info)

# you can use up to 10 timers at the same time
_t0 = [None] * 10

def tic(msg=None, timer_id=0):
    global _t0

    if msg != None:
    	print msg

    timer_id = abs(int(timer_id) % 10) # sanitize input
    _t0[timer_id] = time.time()

def toc(timer_id=0):
    global _t0
    
    timer_id = abs(int(timer_id) % 10) # sanitize input
    if _t0[timer_id] == None:
        print "Please call tic() to start timer first"
        return

    _print_time_in_miliseconds(_t0[timer_id], time.time())
    _t0[timer_id] = None

#--------------------------------

# misc

# returns the number of lines of the given file
# if something goes wrong, returns a string that says 'unknown'
def get_linecount(fname, timeit=False):
	timer_id = 9

	print 'Counting total lines of the file %s...' % fname
	if timeit:
		tic(timer_id=timer_id)
	
	proc = Popen(['wc', '-l', fname], stdout=PIPE, stderr=PIPE)
	out, err = proc.communicate()
	if proc.returncode != 0:
		out = "'unknown'"
	else:
		out = int(out.strip().split(' ')[0])

	if timeit:
		toc(timer_id=timer_id)

	return out
