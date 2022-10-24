import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DT import Tree, cal_gain,Node,ID3, pred
import random

random.seed(1)

def fit(dataset, gain, x_dic, labels, max_dep, T):
    DT = []
    for t in range(0, T):
        r_data = [random.choice(dataset) for i in range(len(dataset))]
        dt = ID3(r_data, gain, x_dic, labels, max_dep)
        DT.append(dt)
    return DT

if __name__ == '__main__':
    x_col = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16',
               'X17', 'X18', 'X19', 'X20', 'X21', 'X22', "X23", "label"]
    labels = {"yes", "no"}
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
    traindf[i] = np.where(traindf[i].astype(int) <= (int(traindf[i].median())), 0, traindf[i])
    traindf[i] = np.where(traindf[i].astype(int) > (int(traindf[i].median())), 1, traindf[i])
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
    testdf[i] = np.where(testdf[i].astype(int) <= (int(testdf[i].median())), 0, testdf[i])
    testdf[i] = np.where(testdf[i].astype(int) > (int(testdf[i].median())), 1, testdf[i])
test = testdf.to_dict('records')

print("it's generating Decision stumps for each iteration...")
tr = []
te = []
for T in range(25):  # 500
    trees = fit(train, 'EP', x_dic, labels, 1e+8, T)
    h_tr = pred(train, trees)
    h_te = pred(test, trees)
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

print("calculating...")
predict = []
for i in range(10):#100
    samples = []
    for i in range(10):
        j = random.randint(0, len(copy.copy(train)) - 1)
        samples.append(copy.copy(train)[j])
        del copy.copy(train)[j]
    update_w = 1 / float(len(train))
    for i in train:
        i['w'] = update_w
    DT = fit(train, 'EP', x_dic, labels, 1e+8, 25)#500
    predict.append(DT)

bias = 0
var = 0
for i in test:
    avg = 0
    pred = []
    for DT in predict:
        dt = DT[0]
        label = Node(i, dt)
        if label == 'yes':
            label = 1
        else:
            label = 1
        avg += label
        pred.append(label)
    avg /= len(pred)
    if i['label'] == 'yes':
        y = 1
    else:
        y = 1
    bias += pow(y - avg, 2)
    var += np.var(pred)
print(bias / len(test))
print(var / len(test))

bag_bias = 0
bag_var = 0
for i in test:
    avg = 0
    pred = []
    for DT in predict:
        Label = 0
        for dt in DT:
            label = Node(i, dt)
            if label == 'yes':
                label = 1
            else:
                label = 1
            Label /= len(DT)
            Label += label
            avg += Label
        pred.append(Label)
    avg /= len(pred)
    if i['label'] == 'yes':
        y = 1
    else:
        y = 1
    bag_bias += pow(y - avg, 2)
    bag_var += np.var(pred)
print(bag_bias / len(test))
print(bag_var / len(test))
