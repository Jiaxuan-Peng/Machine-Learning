#!/usr/bin/env python
# coding: utf-8

# In[99]:


import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from numpy import linalg


# In[100]:


# Set the maximum number of epochs T to 100
# shuffle
# the MAP estimation
def LR_MAP(x, y, v, gamma, d, T, tolerance):
    w= [0]*(x.shape[1])
    m = x.shape[0]
    gamma_0 = gamma
    #learning rate
    for i in range(T):
        x, y = shuffle(x, y,random_state = 1)
        for j in range(len(y)):
            temp = [1/v * i for i in w]
            L = (-m*y[j]*x.iloc[j])/(1+np.exp(-np.dot(w,x.loc[j])*y[j]))+temp
            w = w - gamma*L
            gamma = gamma_0/(1+gamma_0/d*i)
            diff = np.linalg.norm(L)
            #convergence
            if (diff < tolerance):
                return w
    return w


# In[101]:


# the maximum likelihood (ML) estimation
def LR_ML(x, y, v, gamma, d, T, tolerance):
    w= [0]*(x.shape[1])
    w_0 = w
    m = x.shape[0]
    gamma_0 = gamma
    #learning rate
    for i in range(T):
        x, y = shuffle(x, y,random_state = 1)
        for j in range(len(y)):
            L = (-m*y[j]*x.iloc[j])/(1+np.exp(-np.dot(w,x.loc[j])*y[j]))
            w = w - gamma*L
            gamma = gamma_0/(1+gamma_0/d*i)
            diff = np.linalg.norm(L)
            #convergenc
            if (diff < tolerance):
                return w
    return w


# In[102]:


def Prediction_LR(x, y, w):
    y_pred = []
    for j in range(len(x)):
        y_pred_temp =  np.dot(w,x.loc[j])
        if y_pred_temp>0:
            y_pred_temp = 1
            y_pred.append(y_pred_temp)
        else:
            y_pred_temp = -1
            y_pred.append(y_pred_temp)
    error = np.sum(np.abs(y-y_pred))/ 2/len(y)
    return error


# In[103]:


'''
train = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)
'''
train=[]
with open('/home/u1413911/Downloads/Machine-Learning-main/LinearRegression/concrete/train.csv', 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        train.append(terms)

test=[]
with open('/home/u1413911/Downloads/Machine-Learning-main/LinearRegression/concrete/test.csv', 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        test.append(terms)
        
train = pd.DataFrame(train,dtype='float64')
test = pd.DataFrame(test,dtype='float64')

x_train = train.iloc[:,0:4]
x_train.insert(0, 'x_0', [1]*(x_train.shape[0]))#b
y_train = train.iloc[:,-1]
y_train.loc[y_train == 0] = -1 # y = [-1,1]
pd.options.mode.chained_assignment = None  # default='warn'

x_test = test.iloc[:,0:4]
x_test.insert(0, 'x_0', [1]*(x_test.shape[0]))#b
y_test = test.iloc[:,-1]
y_test.loc[y_test == 0] = -1
pd.options.mode.chained_assignment = None  # default='warn'


# In[106]:


variance = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
for v in variance:
    w1 = LR_MAP(x_train, y_train, v, gamma=0.01, d=0.01, T=100, tolerance=1e-5)
    #print(', '.join('{:.3f}'.format(f) for f in w1))
    error_train = Prediction_LR (x_train, y_train, w1)
    error_test = Prediction_LR (x_test, y_test, w1)
    print("variance",v,"error_train:{:.3f}".format(error_train),"error_test:{:.3f}".format(error_test))


# In[105]:


variance = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
for v in variance:
    w2 = LR_ML(x_train, y_train, v, gamma=0.01, d=0.01, T=100, tolerance=1e-5)
    #print(', '.join('{:.3f}'.format(f) for f in w2))
    error_train = Prediction_LR (x_train, y_train, w2)
    error_test = Prediction_LR (x_test, y_test, w2)
    print("variance",v,"error_train:{:.3f}".format(error_train),"error_test:{:.3f}".format(error_test))


# In[ ]:




