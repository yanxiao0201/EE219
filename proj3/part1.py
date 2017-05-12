from common import *
import numpy as np
import os, subprocess

#R = pd.pivot_table(df, index='User', values='Rating', columns='Movie')
#R.to_csv(MATLAB_INFILE, index=False, header=False)
create_matlab_input(R)

# the rest completed in matlab
#print "Executing matlab code..."
#os.chdir('matlab')
#subprocess.call(['matlab', '-nodesktop', 'decmp.m'])
execute_matlab_code()
