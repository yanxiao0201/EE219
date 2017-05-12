from common import *
import util

# one iteration

X = np.array(R)
X, X_test = util.train_test_split(X, train_size=0.9, random_state=util.randint())

#print X, X_test

#create_matlab_input(X)
#execute_matlab_code()
