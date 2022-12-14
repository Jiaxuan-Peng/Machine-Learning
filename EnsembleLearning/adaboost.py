import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DT import Tree, cal_gain,Node,ID3

def fit(dataset, gain, x_dic, labels, T):
    DT = []
    alphas = []
    for t in range(0, T):
        dt = ID3(dataset, gain, x_dic, labels, 2)
        DT.append(dt)
        error = 0
        for i in dataset:
            node = Node(i, dt)
            if node != i['label']:
                error += i['w']
        alpha = 0.5 * math.log((1 - error) / error)
        alphas.append(alpha)
        norm = 0
        for i in dataset:
            node = Node(i, dt)
            if node != i['label']:
                w_new = i['w'] * math.exp(alpha)
            else:
                w_new = i['w'] * math.exp(-alpha)
            i['w'] = w_new
            norm += w_new
        for i in dataset:
            i['w'] /= norm
    return DT, alphas


def pred(dataset, DT, alphas):
    r = 0
    for i in dataset:
        pred = 0
        for dt, alpha in zip(DT, alphas):
            node = Node(i, dt)
            node = 1 if node == 'yes' else -1
            pred += node * alpha
        if i['label'] == 'yes' and pred > 0:
            r += 1
        if i['label'] == 'no' and pred < 0:
            r += 1
    return r / (len(dataset))

def adaboost(train, test, gain, x_dic, labels, T):
    e1 = []
    e2 = []

    for i in train:
        i['w'] = 1 / float(len(train))
    for i in test:
        i['w'] = 1 / float(len(test))
    for t in range(0, T):
        dt = ID3(train, gain, x_dic, labels, 1)
        # calculate votes
        err1 = 0
        for i in train:
            node = Node(i, dt)
            if node != i['label']:
                err1 += i['w']
        err_train = err1
        err2 = 0
        for i in test:
            node = Node(i, dt)
            if node != i['label']:
                err2 += i['w']
        err_test = err2
        e1.append(err_train)
        e2.append(err_test)
        alpha = 0.5 * math.log((1 - err_train) / err_train)
        norm = 0
        for i in train:
            if node != i['label']:
                w_new = i['w'] * math.exp(alpha)
            else:
                w_new = i['w'] * math.exp(-alpha)
            i['w'] = w_new
            norm += w_new
        for i in train:
            i['w'] /= norm
    return e1, e2

if __name__ == '__main__':

    x_dic = {'age': [0, 1],'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
        'marital': ['married','divorced','single'],'education': ['unknown', 'secondary', 'primary', 'tertiary'],
        'default': ['yes', 'no'],'balance': [0, 1],'housing': ['yes', 'no'],'loan': ['yes', 'no'],
        'contact': ['unknown', 'telephone', 'cellular'],'day': [0, 1],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'duration': [0, 1], 'campaign': [0, 1], 'pdays': [0, 1], 'previous': [0, 1], 'poutcome': ['unknown', 'other', 'failure', 'success']}
    x_dic_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    x_col = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays','previous' , 'poutcome', 'label']
    labels = {'yes', 'no'}

    train = []
    y_train = []
    test = []
    y_test = []

    with open('/home/u1413911/Downloads/Machine-Learning-main/DecisionTree/bank/train.csv', 'r') as f:
        for line in f:
            x_val = {}
            term = line.strip().split(',')
            for i in range(len(term)):
                x_val[x_col[i]] = term[i]
            train.append(x_val)
            y_train.append(term[-1])

    #use the median to convert the value
    traindf = pd.DataFrame(train)
    for i in x_dic_numeric:
        traindf[i] = np.where(traindf[i].astype(int) <=(int(traindf[i].median())), 0, traindf[i])
        traindf[i] = np.where(traindf[i].astype(int) >(int(traindf[i].median())), 1, traindf[i])
    train = traindf.to_dict('records')

    #for i in traindf:
    #    print(set(train[i].values()))

    with open('/home/u1413911/Downloads/Machine-Learning-main/DecisionTree/bank/test.csv', 'r') as f:
        for line in f:
            x_val = {}
            term = line.strip().split(',')
            for i in range(len(term)):
                x_val[x_col[i]] = term[i]
            test.append(x_val)
            y_test.append(term[-1])
            
    #use the median to convert the value
    testdf = pd.DataFrame(test)
    for i in x_dic_numeric:
        testdf[i] = np.where(testdf[i].astype(int) <=(int(testdf[i].median())), 0, testdf[i])
        testdf[i] = np.where(testdf[i].astype(int) >(int(testdf[i].median())), 1, testdf[i])
    test = testdf.to_dict('records')


print("it's generating Decision stumps for each iteration...")
e1, e2 = adaboost(train, test, 'EP', x_dic, labels, 25)  # 500
t = [i + 1 for i in range(0, 25)]  # 500
fig, ax = plt.subplots()
ax.plot(t, e2, label='train error', c='green')
ax.plot(t, e1, label='test error', c='red')
ax.legend()
ax.set_xlabel('iteration')
ax.set_ylabel('error')
plt.show()

print("it's generating Decision stumps for each iteration...")
tr = []
te = []
for T in range(25):  # 500
    DT, alphas = fit(train, 'EP', x_dic, labels, T)
    h_tr = pred(train, DT, alphas)
    h_te = pred(test, DT, alphas)
    h_x = 1 - h_tr
    h_y = 1 - h_te
    tr.append(h_x)
    te.append(h_y)
    update_w = 1 / float(len(train))
    for i in train:
        i['w'] = update_w

x1 = [x for x in range(25)]  # 500
fig, ax = plt.subplots()
ax.plot(x1, tr, label='train error', c='red')
ax.plot(x1, te, label='test error', c='green')
ax.legend()
ax.set_xlabel('iteration')
ax.set_ylabel('error')
plt.show()
