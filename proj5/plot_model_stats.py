from matplotlib import pyplot as plt
import os, sys
import util

flist = os.listdir('.')
flist = [f for f in flist if 'model_stats' in f and '.txt' in f] # grep
print flist
if len(flist) == 0:
	raise Exception("Model stats file not found!")

n_hour_window = []
r_squared = []
r_squared_adj = []
hashtag = util.get_hashtags(flist[0]) 	# assumes same hashtag...

for fname in flist:
	for line in open(fname, 'r'):
		line = line.strip()
		tokens = line.split(':')
		if len(tokens) < 2 or len(tokens[1]) == 0:
			continue
		opt = tokens[0]
		val = float(tokens[1])

		if opt == 'n-hour-window':
			n_hour_window.append(val)
		elif opt == 'R-squared':
			r_squared.append(val)
		elif opt == 'Adjusted R-squared':
			r_squared_adj.append(val)

line1, = plt.plot(n_hour_window, r_squared)
line2, = plt.plot(n_hour_window, r_squared_adj)
plt.legend([line1, line2], ['R-squared', 'Adjusted R-squared'])
plt.xlabel('n-hour-window')
plt.title(hashtag)
plt.savefig('r_squared_%s.png' % hashtag)
plt.show()
