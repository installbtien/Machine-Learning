# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:49:05 2021

@author: installbtien

HW2: Binary Classification by generative model
"""

import numpy as np

X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
    
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
    
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)


# Functions
def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1,-1)
        X_std = np.std(X[:, specified_column], 0).reshape(1,-1)
    
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio=0.25):
    
    train_size = int(len(X) * (1-dev_ratio))
    
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def _sigmoid(z):
    
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - 1e-8)

def _f(X, w, b):
    # Logistic regression function, parameterized by w and b
    
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    return np.round(_f(X, w, b)).astype(np.int)

def _accuracy(Y_pred, Y_label):
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

#------------------------------------------------------------------------------------
# Normalizing training data and testing data
X_train, X_mean, X_std = _normalize(X_train)
X_test, _, _ = _normalize(X_test, train=False, X_mean=X_mean, X_std=X_std)

# Split the data
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=0.1)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))

#------------------------------------------------------------------------------------
# Mean and Covariance
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y==0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y==1])

mean_0 = np.mean(X_train_0, axis=0)
mean_1 = np.mean(X_train_1, axis=0)

cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]

for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0])  / (X_train_0.shape[0] + X_train_1.shape[0])

#------------------------------------------------------------------------------------
# Compute w and b
u, s, v = np.linalg.svd(cov, full_matrices=False)
cov_inv = np.matmul(v.T * 1 / s, u.T)

w = np.dot(cov_inv, mean_0 - mean_1)
b = -0.5 * np.dot(mean_0, np.dot(cov_inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(cov_inv, mean_1)) + np.log(float(X_train_0.shape[0] / X_train_1.shape[0]))

# Compute accuracy on training set
Y_train_pred = 1 - _predict(X_train, w, b)
print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))

#------------------------------------------------------------------------------------
# Predict testing data
predictions = 1 - _predict(X_test, w, b)
with open('./data/output_generative.csv', 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
    
# Print out the most  significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)

for i in ind[0:10]:
    print(features[i], w[i])