import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DT_credit import Tree, cal_gain,Node,ID3
from adaboost import fit, pred, adaboost

if __name__ == '__main__':
    x_col = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16',
               'X17', 'X18', 'X19', 'X20', 'X21', 'X22', "X23", "label"]
    labels = {1, 0}
    x_dic_numeric = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', "X23"]
    x_dic = {'X1': [0, 1],'X2': [1, 2],'X3': [0, 1, 2, 3, 4, 5, 6],'X4': [0, 1, 2, 3],'X5': [0, 1],'X6': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             'X7': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],'X8': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],'X9': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             'X10': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],'X11': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],'X12': [0, 1],'X13': [0, 1],'X14': [0, 1],
             'X15': [0, 1],'X16': [0, 1],'X17': [0, 1],'X18': [0, 1],'X19': [0, 1],'X20': [0, 1],'X21': [0, 1],'X22': [0, 1],"X23": [0, 1]}

'''
data = pd.read_csv('credit.csv', header=None)
train_data = data.iloc[:24000]
test_data = data.iloc[24000:]
train_data.to_csv('train.csv', header=None, index=False)
test_data.to_csv('test.csv', header=None, index=False)
'''
train = []
y_train = []
test = []
y_test = []

with open('train.csv', 'r') as f:
    for line in f:
        x_val = {}
        term = line.strip().split(',')
        for i in range(len(term)):
            x_val[x_col[i]] = term[i]
        train.append(x_val)
        y_train.append(term[-1])

# use the median to convert the value
traindf = pd.DataFrame(train)
traindf = traindf.drop([0, 1])

for i in x_dic_numeric:
    traindf[i] = np.where(traindf[i].astype(int) <= (int(float(traindf[i].median()))), 0, traindf[i])
    traindf[i] = np.where(traindf[i].astype(int) > (int(float(traindf[i].median()))), 1, traindf[i])
train = traindf.to_dict('records')

# for i in traindf:
#    print(set(train[i].values()))

with open('test.csv', 'r') as f:
    for line in f:
        x_val = {}
        term = line.strip().split(',')
        for i in range(len(term)):
            x_val[x_col[i]] = term[i]
        test.append(x_val)
        y_test.append(term[-1])

# use the median to convert the value
testdf = pd.DataFrame(test)
testdf = testdf.drop([0, 1])

for i in x_dic_numeric:
    testdf[i] = np.where(testdf[i].astype(int) <= (int(float(testdf[i].median()))), 0, testdf[i])
    testdf[i] = np.where(testdf[i].astype(int) > (int(float(testdf[i].median()))), 1, testdf[i])
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
