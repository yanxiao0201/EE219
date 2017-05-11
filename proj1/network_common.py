#import numpy as np
#from matplotlib import pyplot as plt
#from sklearn import preprocessing
#from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.metrics import mean_squared_error
import pandas as pd
#import csv

#network_pre = []
#dow_dict = { 'Monday' : 1,
#			 'Tuesday' : 2,
#			 'Wednesday' : 3,
#			 'Thursday' : 4,
#			 'Friday' : 5,
#			 'Saturday' : 6,
#			 'Sunday' : 7 }
#with open('network_backup_dataset.csv', 'rb') as f:
#    reader = csv.reader(f, delimiter=',')
#    for csv_row in reader:
#    	if csv_row[1] in dow_dict:
#    		csv_row[1] = dow_dict[csv_row[1]]
#        network_pre.append(csv_row)
#
##Get rid of the first row
#network_data = network_pre[1:]
#    
##Label encoding on features
#lb = preprocessing.LabelBinarizer()
#
#week = np.array([int(row[0]) for row in network_data])
#day_of_week = np.array(lb.fit_transform([row[1] for row in network_data]))
#start_time = np.array([int(row[2]) for row in network_data])
#workflow = np.array(lb.fit_transform([row[3] for row in network_data]))
#file_name = np.array(lb.fit_transform([row[4] for row in network_data]))
#size = np.array([float(row[5]) for row in network_data])
#backup_time = np.array([int(row[6]) for row in network_data])
#other = np.column_stack((week, day_of_week, start_time, workflow, file_name, backup_time))

df = pd.read_csv('network_backup_dataset.csv')
df.columns = ['WEEK', 'DAY', 'ST', 'WRKF', 'FILE', 'SIZE', 'BACK']
week = df['WEEK']
day_of_week = pd.get_dummies(df['DAY'])
start_time = df['ST']
workflow = pd.get_dummies(df['WRKF'])
file_name = pd.get_dummies(df['FILE'])
size = df['SIZE']
backup_time = df['BACK']
frames = [week, day_of_week, start_time, workflow, file_name, backup_time]
other = pd.concat(frames, axis=1)

