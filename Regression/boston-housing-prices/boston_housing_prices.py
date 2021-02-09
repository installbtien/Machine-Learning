# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 03:25:11 2021

@author: installbtien

Boston Housing Prices by regression
predict MEDV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# https://www.kaggle.com/c/hmboost/data?select=train.csv

df = pd.read_csv('train.csv')
df = df.iloc[:, 1:]
data = df.to_numpy()
x = data[:, :13]
y = np.empty([404,1])

for i in range(404):
    y[i][0] = data[i][13]

# -----------------------------------------------------
# feature scaling
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)

for i in range(404):
    for j in range(13):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# -----------------------------------------------------
# y = w*x + b

w = np.zeros([14,1])
x = np.concatenate((np.ones([404,1]), x), axis=1).astype(float)
# x_train = x[:303 , :]
# x_val = x[303: , :]

lr = 1
iteration = 10000
sum_grad2 = np.zeros([14,1]) #????
rms = np.empty([iteration, 1])

# loss function: sum( y - x*w ) ^ 2
# gradient = 2 * x *( x*w - y )
for i in range(iteration):
    grad = 2 * np.dot(x.transpose(), np.dot(x,w) - y)
    sum_grad2 += grad ** 2
    w -= lr / np.sqrt(sum_grad2) * grad
    loss = np.sum((np.dot(x,w) - y) ** 2)
    rms[i][0] = np.sqrt(loss / 14)
    
# plot rms-iteration
plt.plot(range(iteration), rms)
plt.xlabel('iteration', fontsize=16)
plt.ylabel('RMS', fontsize=16)
plt.show()

# -----------------------------------------------------
df_test = pd.read_csv('test.csv')
df_test = df_test.iloc[:, 1:]
x_test = df_test.to_numpy()

mean_test = np.mean(x_test, axis=0)
std_test = np.std(x_test, axis=0)

for i in range(102):
    for j in range(13):
        if std_test[j] != 0:
            x_test[i][j] = (x_test[i][j] - mean_test[j]) / std_test[j]
            
x_test = np.concatenate((np.ones([102,1]), x_test), axis=1).astype(float)
y_predict = np.dot(x_test, w)

# -----------------------------------------------------
import csv
with open('predict.csv', mode='w', newline='') as predict_file:
    csv_writer = csv.writer(predict_file)
    head = ['index', 'target']
    csv_writer.writerow(head)
    
    for i in range(102):
        row = [i , y_predict[i][0]]
        csv_writer.writerow(row)
