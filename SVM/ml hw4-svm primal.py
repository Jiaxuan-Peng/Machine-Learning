#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle


# In[37]:


#implement SVM in the primal domain with stochastic sub-gradient descent
#Set the maximum epochs T to 100
def SVM_primal_gamma1(x, y, C, gamma, a, T):
    print("Calculating...")
    w= [0]*(x.shape[1])
    N = (x.shape[0])
    gamma_0 = gamma
    #learning rate
    for i in range(T):
        x, y = shuffle(x, y,random_state = 1)
        for j in range(len(y)):
            if np.sum(y[j]*np.transpose(w)*x.loc[j].values) <=1:
                #update weight_vector
                w = [i * (1-gamma) for i in w]
                w = [a + b for a, b in zip(w, gamma * C*N*y[j]*x.loc[j])] #w -gamma w + gammaC N yi xi
            else:
                w = [i * (1-gamma) for i in w]
            gamma = gamma_0/(1+gamma_0/a*i)
    return w


# In[38]:


#implement SVM in the primal domain with stochastic sub-gradient descent
#Set the maximum epochs T to 100
def SVM_primal_gamma2(x, y, C, gamma,T):
    print("Calculating...")
    w= [0]*(x.shape[1])
    N = (x.shape[0])
    gamma_0 = gamma
    #learning rate
    for i in range(T):
        x, y = shuffle(x, y,random_state = 1)
        for j in range(len(y)):
            if np.sum(y[j]*np.transpose(w)*x.loc[j].values) <=1:
                #update weight_vector
                w = [i * (1-gamma) for i in w]
                w = [a + b for a, b in zip(w, gamma * (C*N*y[j]*x.loc[j]))] #w -gamma w + gammaC N yi xi
            else:
                w = [i * (1-gamma) for i in w]
            gamma = gamma_0/(1+i)
    return w


# In[39]:


#Prediction: sgn(wTx)
def Prediction_SVM_primal(x, y, w):
    print("Calculating...")
    y_pred = []
    for j in range(len(x)):
        y_pred_temp =  np.sign(np.sum(w*x.loc[j].values))
        y_pred.append(y_pred_temp)
    error = np.sum(np.abs(y-y_pred))/len(y)
    return error


# In[40]:


train = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)

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


# In[43]:


#Try the hyperpa-rameter C from {100/873 , 500/873,700/873}
for C in [100/873 , 500/873,700/873]:
    w1 = SVM_primal_gamma1(x_train, y_train, C, gamma=0.01, a=0.001,T=100)
    print(', '.join('{:.3f}'.format(f) for f in w1))
    error_train = Prediction_SVM_primal (x_train, y_train, w1)
    print("error_train:{:.3f}".format(error_train))
    error_test = Prediction_SVM_primal (x_test, y_test, w1)
    print("error_test:{:.3f}".format(error_test))


# In[44]:


#Try the hyperpa-rameter C from {100/873 , 500/873,700/873}
for C in [100/873 , 500/873,700/873]:
    w2 = SVM_primal_gamma2(x_train, y_train, C, gamma=0.01,T=100)
    print(', '.join('{:.3f}'.format(f) for f in w2))
    error_train = Prediction_SVM_primal (x_train, y_train, w2)
    print("error_train:{:.3f}".format(error_train))
    error_test = Prediction_SVM_primal (x_test, y_test, w2)
    print("error_test:{:.3f}".format(error_test))


# In[ ]:




