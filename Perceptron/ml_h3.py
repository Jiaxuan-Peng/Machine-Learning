#!/usr/bin/env python
# coding: utf-8

# In[263]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle


# In[264]:


#Implement the standard Perceptron. Set the maximum number of epochs T to 10
def Standard_Perceptron(x, y, learn_rate, T):
    w= [0]*(x.shape[1])
    for i in range(T):
        x, y = shuffle(x, y,random_state = 1)
        for j in range(len(y)):
            if np.sum(y[j]*np.transpose(w)*x.loc[j].values) <=0:
                #update weight_vector
                w = [a + b for a, b in zip(w, learn_rate * y[j] * x.loc[j])]#w + ryx
    return w


# In[265]:


#the list of the distinct weight vectors and their counts — the number of correctly predicted training examples
def voted_Perceptron(x, y, learn_rate, T):
    pd.options.mode.chained_assignment = None  # default='warn'
    w= [0]*(x.shape[1])
    m= 0
    c=1
    w_all = []
    c_all = []
    for i in range(T):
        x, y = shuffle(x, y,random_state = 1)
        for j in range(len(y)):
            if np.sum(y[j]*np.transpose(w)*x.loc[j].values) <=0:
                #update weight_vector
                w = [a + b for a, b in zip(w, learn_rate * y[j] * x.loc[j])]#w + ryx
                w_all.insert(m,w)
                c_all.insert(m,c)                
                m=m+1
                c=1                                
            else:
                c=c+1
    return w_all, c_all


# In[266]:


def Average_Perceptron(x, y, learn_rate, T):
    w= [0]*(x.shape[1])
    a= [0]*(x.shape[1])
    for i in range(T):
        x, y = shuffle(x, y,random_state = 1)
        for j in range(len(y)):
            if np.sum(y[j]*np.transpose(w)*x.loc[j].values) <=0:
                #update weight_vector
                w = [first + second for first, second in zip(w, learn_rate * y[j] * x.loc[j])]#w + ryx
                a = [first + second for first, second in zip(w, a)]#a+w
    return a


# In[267]:


#Prediction: sgn(aTx)
def Prediction_Average (x, y, a):
    y_pred = []
    for j in range(len(x)):
        y_pred_temp =  np.sign(np.sum(a*x.loc[j].values))
        y_pred.append(y_pred_temp)
    error = np.sum(np.abs(y-y_pred))/len(y)
    return error


# In[268]:


#Prediction: sgn(wTx)
def Prediction_Standard (x, y, w):
    y_pred = []
    for j in range(len(x)):
        y_pred_temp =  np.sign(np.sum(w*x.loc[j].values))
        y_pred.append(y_pred_temp)
    error = np.sum(np.abs(y-y_pred))/len(y)
    return error


# In[269]:


#Prediction: sgn(sum(c*sgn(wTx)))
def Prediction_voted (x, y, w, c):
    print("calculating...")
    y_pred = []
    for j in range(len(x)):
        y_pred_temp = []
        for i in range(len(w)):
            temp = c[i]*np.sign(np.sum(np.transpose(w[i])*x.loc[j].values))
            y_pred_temp.append(temp)
        y_pred_temp1 =  np.sign(np.sum(y_pred_temp))
        y_pred.append(y_pred_temp1)
    error = np.sum(np.abs(y-y_pred))/len(y)
    return error


# In[270]:


train = pd.read_csv('/home/u1413911/Downloads/train.csv', header=None)
test = pd.read_csv('/home/u1413911/Downloads/test.csv', header=None)

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


# In[271]:


#Report your learned weight vector, and the average prediction error on the test dataset
w_standard = Standard_Perceptron(x_train, y_train, 0.01, 10)
print(', '.join('{:.3f}'.format(f) for f in w_standard))
error_standard = Prediction_Standard (x_test, y_test, w_standard)
print(error_standard)


# In[272]:


#Report your learned weight vector, and the average prediction error on the test dataset
w_voted, c_voted = voted_Perceptron(x_train, y_train, 0.01, 10)
for i in range(len(w_voted)):
    print(['%.3f' % n for n in w_voted[i]])
print((c_voted))
error_voted = Prediction_voted(x_test, y_test, w_voted, c_voted)
print(error_voted)


# In[273]:


#Report your learned weight vector, and the average prediction error on the test dataset
w_average = Average_Perceptron(x_train, y_train, 0.01, 10)
print(', '.join('{:.3f}'.format(f) for f in w_average))
error_average =  Prediction_Average(x_test, y_test, w_average)
print(error_average)


# In[274]:


#the relationship between vote and average
sum = []
for j in range(len(w_voted[0])):
    a = 0
    for i in range(len(w_voted)):
        a = a+w_voted[i][j]
    sum.append(a)
print(sum)


# In[ ]:




