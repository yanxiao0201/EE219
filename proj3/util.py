import copy, random, time

def _gen_rand_idx_lst(array_size, list_len, seed=None):
	if array_size < 0 or list_len < 0 or array_size < list_len:
		raise Exception("_gen_rand_idx_lst: illegal arguments")

	# heuristics
	h_ratio = 0.15
	h_nsweeps = 3

	random.seed(seed)
	if float(list_len) / array_size <= h_ratio:
		ret = []
		while list_len > 0:
			n = random.randint(0, array_size - 1)
			if n not in ret:
				ret.append(n)
				list_len -= 1
		return ret
	else:
		# requested list length close to array size
		# use a different method: shuffle and truncate
		shuffle_list = []
		idx_lst = list(xrange(array_size))
		
		for i in xrange(array_size * h_nsweeps):
			r = random.randint(0, array_size - 1)
			shuffle_list.append(r)
		
		for idx, r in enumerate(shuffle_list):
			i = idx % len(idx_lst)
			# swap the random index with in-order index
			t = idx_lst[r]
			idx_lst[r] = idx_lst[i]
			idx_lst[i] = t
		return idx_lst[:list_len]

# returns modified X and X_test which is a dict that
# maps row # to row value (note that it is not user_id)
def train_test_split(X, train_size=0.9, random_state=42):
	train_size = 0.9
	m, n = X.shape
	m_train = int(m * train_size)
	m_test = m - m_train

	X_test = {}
	index_list = _gen_rand_idx_lst(m, m_test, random_state)
	for i in index_list:
		#X_test.append(list(X[i]))
		X_test[i] = copy.copy(X[i])
		X[i] = float('nan')

	return X, X_test

def randint():
	random.seed()
	return random.random() * 2**30
