#import numpy as np
#from matplotlib import pyplot as plt
#from sklearn import preprocessing
#from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.metrics import mean_squared_error
import pandas as pd

features = pd.read_csv('housing_data.csv')
features.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
target = features['MEDV']
del features['MEDV']


