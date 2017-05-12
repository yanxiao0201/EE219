import pandas as pd
import numpy as np
import sys

fname = 'ml-latest-small/ratings.csv'

if len(sys.argv) > 1:
	fname = sys.argv[1]

df = pd.read_csv(fname)
df.columns = ['User', 'Movie', 'Rating', 'Time']
del df['Time']
R = pd.pivot_table(df, index='User', values='Rating', columns='Movie')

R.to_csv('matlab/out.csv', index=False, header=False)
