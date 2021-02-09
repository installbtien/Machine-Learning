# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 03:37:34 2021

@author: installbtien

HW1-predict PM 2.5 by regression(gradient descent, adagrad)
"""

import pandas as pd
import numpy as np

data = pd.read_csv('train.csv', encoding='big5')

# 只需取第三列以後的data
data = data.iloc[:,3:]
# NR = 0
data[data == 'NR'] = 0

raw_data = data.to_numpy()


month_data = {}
for month in range(12):
    
    # 18種空氣成分 一個月18*30筆
    temp_data = np.empty([18,480])
    
    for day in range(20):
        # 每天有24份數據
        temp_data[:, 24*day  : 24*(day+1)] = raw_data[18 * (20*month+day) : 18 * (20*month+day+1)]
    
    month_data[month] = temp_data


#-----------------------------------------------------------
# 每10小時取出一筆training data
# 每個月連續時間有20*24=480小時，10小時一個data，可分為471筆data，12個月共12*471筆
# 每一筆data，前9小時為input，最後一小時的PM 2.5為output，因此input矩陣為18*9

# x是input
x = np.empty([12*471,18*9], dtype=float)
# y是output
y = np.empty([12*471,1], dtype=float)

for month in range(12):
    
    for i in range(471):
        
        temp_data = np.empty([18,10])
        temp_data = month_data[month][:, i:i+10]
        
        x[471*month+i, :] = temp_data[:, 0:9].reshape(1,-1)
        y[471*month+i] = temp_data[9,9]
        

#-----------------------------------------------------------
# feature scale: normalization
x2 = x.copy()

mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)

for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]


#-----------------------------------------------------------
# 另一種normalization: 18*9的feature中，每9個都是同一種物質，因此可以9個一組求mean, std

mean_x2 = np.zeros(18)
std_x2 = np.zeros(18)

for i in range(18):
    mean_x2[i] = np.mean([x2[:, 9*i:9*(i+1)]])
    std_x2[i] = np.std([x2[:, 9*i:9*(i+1)]])
    
for i in range(18):
    if std_x2[i] != 0:
        x2[:, 9*i:9*(i+1)] = (x2[:, 9*i:9*(i+1)] - mean_x2[i]) / std_x2[i]
        

#-----------------------------------------------------------
# 為了衡量testing  data 的bias對結果的影響，切出一塊validation data
import math
x_train = x[ : math.floor(len(x)*0.8), :]
y_train = y[ : math.floor(len(y)*0.8), :]

x_valid = x[math.floor(len(x)*0.8): , :]
y_valid = y[math.floor(len(y)*0.8): , :]

#-----------------------------------------------------------
# linear regression: y = w1*x1 + w2*x2 + ... + b
# 原本18*9加上常數，加1維
import matplotlib.pyplot as plt

dim = 18*9 + 1
w = np.zeros([dim,1])
# 用np.concatenate在x加上一個常數feature=1，用1的列向量，用axis=1連接
x = np.concatenate((np.ones([12*471,1]), x), axis=1).astype(float)

lr = 0.1
iteration = 100000

adagrad = np.zeros([dim, 1])
eps = 0.000000001
rms = np.empty([iteration, 1])

# loss function: sum(y-(b+w*x))^2
# gradient = 2*x*((b+w*x)-y)
for i in range(iteration):
    
    grad = 2 * np.dot(x.transpose(), np.dot(x, w)-y)
    adagrad += grad ** 2
    w -= lr / np.sqrt(adagrad + eps) * grad
    loss = np.sum((y - np.dot(x ,w)) ** 2)
    rms[i][0] = np.sqrt(loss / (471*12))

# rms[99999][0] 5.6796521875580375 效果較好

# save w
np.save('weight_adgrad.npy', w)

plt.plot(range(len(rms)), rms)
plt.show()

#-----------------------------------------------------------
# x2

# w2 = np.zeros([dim,1])
# x2 = np.concatenate((np.ones([12*471,1]), x2), axis=1).astype(float)

# lr = 0.1
# iteration = 100000

# adagrad = np.zeros([dim, 1])
# eps = 0.0000000001
# rms = np.empty([iteration, 1])

# for i in range(iteration):
    
#     grad = 2 * np.dot(x2.transpose(), np.dot(x2, w2)-y)
#     adagrad += grad ** 2
#     w2 -= lr / np.sqrt(adagrad + eps) * grad
#     loss = np.sum((y - np.dot(x2 ,w2)) ** 2)
#     rms[i][0] = np.sqrt(loss / (471*12))

# rms[99999][0] 5.679654710309812 效果較差

# # save w
# np.save('weight_adgrad2.npy', w2)

# plt.plot(range(len(rms)), rms)
# plt.show()

#----------------------------------------------------------

testdata = pd.read_csv('test.csv', header=None, encoding='big5')
testdata = testdata.iloc[:, 2:]
testdata[testdata == 'NR'] = 0
testdata = testdata.to_numpy()
x_test = np.empty([240, 18*9], dtype=float)

for i in range(240):
    x_test[i, :] = testdata[18*i : 18*(i+1), :].reshape(1,-1)

for i in range(len(x_test)):
    for j in range(len(x_test[0])):
        if std_x[j] != 0:
            x_test[i][j] = (x_test[i][j] - mean_x[j]) / std_x[j]

x_test = np.concatenate((np.ones([240,1]), x_test), axis=1).astype(float)
print(x_test)

#----------------------------------------------------------

y_predict = np.dot(x_test, w)
print(y_predict)

#----------------------------------------------------------

import csv
with open('predict_adagrad.csv', mode='w', newline='') as predict_file:
    csv_writer = csv.writer(predict_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    
    for i in range(240):
        row = ['id_' + str(i), y_predict[i][0]]
        csv_writer.writerow(row)
        print(row)
