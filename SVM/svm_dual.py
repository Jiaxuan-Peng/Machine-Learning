#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from scipy.optimize import minimize


# In[32]:


def kernel(x,y,gamma):
    return np.exp(-1*np.linalg.norm(x-y, 2) / gamma)

'''
def Kernel(x,y,gamma):
    for i in range(len(x)):
        for j in range(len(y)):
            Kernel=kernel(x[i],y[j],gamma)
    return Kernel
'''


# In[34]:


#implement SVM in the dual domain
def SVM_dual_kernel(x, y, C,gamma):
    N = x.shape[0]
    x0 = np.zeros(N)#initial guess
    bnds = [(0, C)] * N#boundary
    #writing down the objective in terms of the matrix and vector operations
    def fun(alpha):#objective function
        return 1/2*alpha.dot(alpha.dot(kernel(x,x,gamma) * np.dot(y, y.T)))-sum(alpha)
    #treat the Lagrange multipliers that we want to optimize as a vector
    def jac(alpha):#gradient
        return alpha.dot(kernel(x,x,gamma) * np.dot(y, y.T)) - np.ones_like(alpha)
    cons = ({'type': 'eq', 'fun': lambda alpha:  np.dot(alpha, y), 'jac': lambda alpha: y})#constraint
    res = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bnds)#optimization
    alpha = res.x#aphla
    # Recover the feature weights w and the bias b
    w = np.dot(x.T, y * res.x)
    b = np.mean(y - kernel(x,w,gamma))
    return alpha,b


# In[35]:


#Prediction: sgn(wTx)
def Prediction_SVM_dual_kernel(x, y, alpha,b):
    y_pred = []
    for j in range(len(x)):
        y_pred_temp =  np.sign(np.sum(kernel(x,x,gamma*y * alpha[j])+b))
        y_pred.append(y_pred_temp)
    error = np.sum(np.abs(y-y_pred))/len(y)
    return error

'''
def Prediction_SVM_dual_kernel(x, y, alpha,b):
    y_pred = []
    for j in range(len(x)):
        for i in range(len(x)):
            y_pred_temp= alpha[j] * y[j] * kernel(x[j], x[i],0.01)
            y_pred.append(y_pred_temp)
    error = np.sum(np.abs(y-y_pred))/len(y)
    return error
'''


# In[36]:


#implement SVM in the dual domain
def SVM_dual(x, y, C):
    N = x.shape[0]
    x0 = np.zeros(N)#initial guess
    bnds = [(0, C)] * N#boundary
    #writing down the objective in terms of the matrix and vector operations
    def fun(alpha):#objective function
        return 1/2*alpha.dot(alpha.dot(np.dot(x, x.T) * np.dot(y, y.T)))-sum(alpha)
    #treat the Lagrange multipliers that we want to optimize as a vector
    def jac(alpha):#gradient
        return alpha.dot(np.dot(x, x.T) * np.dot(y, y.T)) - np.ones_like(alpha)
    cons = ({'type': 'eq', 'fun': lambda alpha:  np.dot(alpha, y), 'jac': lambda alpha: y})#constraint
    res = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bnds)#optimization
    res.x#aphla
    # Recover the feature weights w and the bias b
    w = np.dot(x.T, y*res.x)
    b = np.mean(y - np.dot(x, w))
    return w,b


# In[37]:


#Prediction: sgn(wTx)
def Prediction_SVM_dual(x, y, w,b):
    y_pred = []
    for j in range(len(x)):
        y_pred_temp =  np.sign(np.sum(w*x.loc[j].values+b))
        y_pred.append(y_pred_temp)
    error = np.sum(np.abs(y-y_pred))/len(y)
    return error


# In[38]:


train = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)

x_train = train.iloc[:,0:4]
#x_train.insert(0, 'x_0', [1]*(x_train.shape[0]))#b
y_train = train.iloc[:,-1]
y_train.loc[y_train == 0] = -1 # y = [-1,1]
pd.options.mode.chained_assignment = None  # default='warn'

x_test = test.iloc[:,0:4]
#x_test.insert(0, 'x_0', [1]*(x_test.shape[0]))#b
y_test = test.iloc[:,-1]
y_test.loc[y_test == 0] = -1
pd.options.mode.chained_assignment = None  # default='warn'


# In[39]:


#Try the hyperpa-rameter C from {100/873 , 500/873,700/873}
for C in [100/873, 500/873, 700/873]:
    print("Calculating...")
    w,b = SVM_dual(x_train, y_train, C)
    print(', '.join('{:.3f}'.format(f) for f in w))
    print("b:{:.3f}".format(b))
    error_train = Prediction_SVM_dual (x_train, y_train, w,b)
    print("error_train:{:.3f}".format(error_train))
    error_test = Prediction_SVM_dual (x_test, y_test, w,b)
    print("error_test:{:.3f}".format(error_test))


# In[40]:


#Try the hyperpa-rameter C from {100/873 , 500/873,700/873}
for gamma in [0.1, 0.5, 1, 5, 100]:
    for C in [100/873, 500/873, 700/873]:
        print("Calculating...")
        alpha,b = SVM_dual_kernel(x_train, y_train, C,gamma)
        print("b:{:.3f}".format(b))
        error_train = Prediction_SVM_dual_kernel (x_train, y_train, alpha,b)
        print("error_train:{:.3f}".format(error_train))
        error_test = Prediction_SVM_dual_kernel (x_test, y_test, alpha,b)
        print("error_test:{:.3f}".format(error_test))
        print('support vectors:', np.sum(alpha != 0))


# In[ ]:


#overlapped support vectors
for gamma in [0.01, 0.1, 0.5]:
    print("Calculating...")
    alpha,b = SVM_dual_kernel(x_train, y_train,gamma, C=500/873)
    print("b:{:.3f}".format(b))
    error_train = Prediction_SVM_dual_kernel (x_train, y_train, alpha,b)
    print("error_train:{:.3f}".format(error_train))
    error_test = Prediction_SVM_dual_kernel (x_test, y_test, alpha,b)
    print("error_test:{:.3f}".format(error_test))
    print('support vectors:', np.sum(alpha != 0))
    support_vectors = np.sum(alpha != 0)
    temp = set(np.zeros(train_x.shape[0]))
    overlapped = [x for x in support_vectors if x not in temp]
    print(overlapped)
    temp = support_vectors


# In[ ]:


#implement SVM in the dual domain
def SVM_dual_kernel(x, y, C,gamma):
    N = x.shape[0]
    x0 = np.zeros(N)#initial guess
    bnds = [(0, C)] * N#boundary
    #writing down the objective in terms of the matrix and vector operations
    def fun(alpha):#objective function
        return 1/2*alpha.dot(alpha.dot(kernel(x,x,gamma) * np.dot(y, y.T)))-sum(alpha)
    #treat the Lagrange multipliers that we want to optimize as a vector
    def jac(alpha):#gradient
        return alpha.dot(kernel(x,x,gamma) * np.dot(y, y.T)) - np.ones_like(alpha)
    cons = ({'type': 'eq', 'fun': lambda alpha:  np.dot(alpha, y), 'jac': lambda alpha: y})#constraint
    res = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bnds)#optimization
    alpha = res.x#aphla
    # Recover the feature weights w and the bias b
    w = np.dot(x.T, y * res.x)
    b = np.mean(y - kernel(x,w,gamma))
    return alpha,b


# In[48]:


# the kernel Perceptron
def kernel_Perceptron(x, y, learn_rate, T,gamma):
    pd.options.mode.chained_assignment = None  # default='warn'
    w= [0]*(x.shape[1])
    m= 0
    c=0
    w_all = []
    c_all = []
    for i in range(T):
        x, y = shuffle(x, y,random_state = 1)
        for j in range(len(y)):
            if y[j]*c*kernel(x.loc[j].values,x.loc[j].values,gamma) <=0:
                #update weight_vector
                w = [a + b for a, b in zip(w, learn_rate * y[j] * x.loc[j])]#w + ryx
                w_all.insert(m,w)
                c_all.insert(m,c)                
                m=m+1
                c=c+1                             
    return w_all, c_all

def Prediction_kernel (x, y, w, c):
    print("calculating...")
    y_pred = []
    for j in range(len(x)):
        y_pred_temp = []
        for i in range(len(y)):
            temp = np.sign(np.sum(y[j]*c*kernel(x,x,gamma)))
            y_pred_temp.append(temp)
        y_pred_temp1 =  np.sign(np.sum(y_pred_temp))
        y_pred.append(y_pred_temp1)
    error = np.sum(np.abs(y-y_pred))/len(y)
    return error


# In[50]:


#Report your learned weight vector, and the average prediction error on the test dataset
for gamma in [0.1, 0.5, 1, 5, 100]:
    w_kernel, c_kernel = kernel_Perceptron(x_train, y_train, 0.01, 10,gamma)
    error_kernel = Prediction_kernel(x_test, y_test, w_kernel, c_kernel)
    print(error_kernel)


# In[ ]:




