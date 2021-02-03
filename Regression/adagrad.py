# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 01:17:45 2021

@author: installbtien
"""

import numpy as np
import matplotlib.pyplot as plt

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
# ydata = b + w * xdata

x = np.arange(-200,-100,1)
y = np.arange(-5,5,0.1)
Z = np.zeros((len(x),len(y)))
X, Y = np.meshgrid(x,y)

for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] += (y_data[n] - b - w*x_data[n])**2
        Z[j][i] = Z[j][i]/len(x_data)

# ydata = b + w * xdata
b = -120 # initial b
w = -4 # initial w
lr = 1 # learning rate
iteration = 10000000

# store initial values for plotting
b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0

# iterations
for i in range(iteration):
    
    b_grad = 0.0
    w_grad = 0.0
    
    for n in range(len(x_data)):
        b_grad -= 2.0*(y_data[n] - b - w*x_data[n])*1.0
        w_grad -= 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
    
        
    lr_b += b_grad ** 2
    lr_w += b_grad ** 2
    
    # update parameters
    b -= lr/np.sqrt(lr_b) * b_grad
    w -= lr/np.sqrt(lr_w) * w_grad
    
    # store parameters for plotting
    b_history.append(b)
    w_history.append(w)
    
# plot the figure
plt.contourf(x,y,Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()

    
    
    
    
    