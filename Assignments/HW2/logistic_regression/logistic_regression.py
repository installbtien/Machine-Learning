# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 23:32:43 2021

@author: installbtien

HW2: Binary Classification by logistic regression
"""

import numpy as np
import matplotlib.pyplot as plt

X_train_fpath = '../data/X_train'
Y_train_fpath = '../data/Y_train'
X_test_fpath = '../data/X_test'

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

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - 1e-8)

def _f(X, w, b):
    # Logistic regression function, parameterized by w and b
    
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    return np.round(_f(X, w, b).astype(np.int))

def _accuracy(Y_pred, Y_label):
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(Y_pred, Y_label):
    loss = -np.dot(Y_label, np.log(Y_pred)) - np.dot(1-Y_label, np.log(1-Y_pred))
    return loss

def _gradient(X, Y_label, w, b):
    y_pred = _f(X,w,b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    
    return w_grad, b_grad

#------------------------------------------------------------------------------------
X_train, X_mean, X_std = _normalize(X_train)
X_test, _, _ = _normalize(X_test, train=False, X_mean=X_mean, X_std=X_std)

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
w = np.zeros((data_dim, ))
b = np.zeros((1, ))

# Mini-batch gradient descent is used here, in which training data are split into several
# mini-batches and each batch is fed into the model sequentially for losses and gradients 
# computation. Weights and bias are updated on a mini-batch basis.

# Once we have gone through the whole training set, the data have to be re-shuffled and 
# mini-batch gradient desent has to be run on it again. We repeat such process until max 
# number of iterations is reached.

max_iter = 10
batch_size = 8
lr = 0.2

train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

step = 1

for epoch in range(max_iter):
    X_train, Y_train = _shuffle(X_train, Y_train)
    
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx*batch_size : (idx+1)*batch_size]
        Y = Y_train[idx*batch_size : (idx+1)*batch_size]
        
        w_grad, b_grad = _gradient(X, Y, w, b)
        
        w -= lr / np.sqrt(step) * w_grad
        b -= lr / np.sqrt(step) * b_grad
        
        step +=1
    
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)
    
    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

# Loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()

#------------------------------------------------------------------------------------
# Predict testing data
predictions = _predict(X_test, w, b)

with open('../data/output_logistic.csv', 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}'.format(i, label))

ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath, 'r') as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)

for i in ind[:10]:
    print(features[i], w[i])
