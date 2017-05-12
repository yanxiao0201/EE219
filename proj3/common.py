import pandas as pd
import numpy as np
import os, subprocess

# parameters/constants
#MATLAB_INFILE='matlab/out.csv'
#MATLAB_EXE_NAME='decmp.m'

# data
df = pd.read_csv('ml-latest-small/ratings.csv')
df.columns = ['User', 'Movie', 'Rating', 'Time']
del df['Time']
R = pd.pivot_table(df, index='User', values='Rating', columns='Movie')

def create_matlab_input(mat):
	fname = 'matlab/out.csv'
	if type(mat) == pd.core.frame.DataFrame:
		mat.to_csv(fname, index=False, header=False)
	elif type(mat) == np.ndarray:
		np.savetxt(fname, mat, delimiter=',')
	else:
		raise Exception("create_matlab_input: unknown input type")

def execute_matlab_code():
	print "Executing matlab code..."
	os.chdir('matlab')
	subprocess.call(['matlab', '-nodesktop', 'decmp.m'])
