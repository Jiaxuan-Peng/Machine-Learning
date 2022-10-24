import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DT import Tree, cal_gain,Node, pred
import random

random.seed(1)

def ID3(dataset, gain, x_dic, labels, max_dep, size):
    if (len(x_dic) == 0) or max_dep == 0:
        c = {}
        for i in dataset:
            label = i['label']
            if label not in c:
                c[label] = 0
            c[label] += i['w']
        node = max(c.keys(), key=lambda key: c[key])
        return Tree(node)
    if (len(labels) == 1):
        return Tree(labels.pop())
    total_gain = {}
    for ln, lv in x_dic.items():
        total_gain_temp = cal_gain(dataset, gain, ln, lv)
        total_gain[ln] = total_gain_temp
        best_attr = max(total_gain.keys(), key=lambda key: total_gain[key])
    root = Tree(best_attr)
    for v in x_dic[best_attr]:
        sub_set = []
        for i in dataset:
            if i[best_attr] == v:
                sub_set.append(i)
        if len(sub_set) == 0:
            c = {}
            for i in dataset:
                label = i['label']
                if label not in c:
                    c[label] = 0
                c[label] += i['w']
            node = max(c.keys(), key=lambda key: c[key])
            root.child[v] = Tree(node)
        else:
            sub_x_dic = copy.copy(x_dic)
            sub_x_dic.pop(best_attr)
            sub_attr = {}
            dict_copy = list(sub_x_dic.keys())
            while len(sub_attr) < size:
                id = random.randint(0, len(dict_copy) - 1)
                dict = dict_copy[id]
                if dict not in sub_attr:
                    sub_attr[dict] = sub_x_dic[dict]
            sub_labels = set()
            for i in sub_set:
                sub_label = i['label']
                if sub_labels not in sub_labels:
                    sub_labels.add(sub_label)
            root.child[v] = ID3(sub_set, gain, sub_attr, sub_labels, max_dep-1, size)
    return root

def fit(dataset, gain, x_dic, labels, max_dep, T, size):
    DT = []
    for t in range(0, T):
        r_data = [random.choice(dataset) for i in range(len(dataset))]
        dt = ID3(r_data, gain, x_dic, labels, max_dep, size)
        DT.append(dt)
    return DT

if __name__ == '__main__':

    x_dic = {'age': [0, 1],
             'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                     'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
             'marital': ['married', 'divorced', 'single'], 'education': ['unknown', 'secondary', 'primary', 'tertiary'],
             'default': ['yes', 'no'], 'balance': [0, 1], 'housing': ['yes', 'no'], 'loan': ['yes', 'no'],
             'contact': ['unknown', 'telephone', 'cellular'], 'day': [0, 1],
             'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
             'duration': [0, 1], 'campaign': [0, 1], 'pdays': [0, 1], 'previous': [0, 1],
             'poutcome': ['unknown', 'other', 'failure', 'success']}
    x_dic_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    x_col = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
             'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
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

    # use the median to convert the value
    testdf = pd.DataFrame(test)
    for i in x_dic_numeric:
        testdf[i] = np.where(testdf[i].astype(int) <= (int(testdf[i].median())), 0, testdf[i])
        testdf[i] = np.where(testdf[i].astype(int) > (int(testdf[i].median())), 1, testdf[i])
    test = testdf.to_dict('records')


print("it's generating Errors vary with the number of random trees...")
tr = []
te = []
for T in range(25):  # 500
    trees = fit(train, 'EP', x_dic, labels, 1e+8, T, 2)
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

sub_attr = {}
dict_copy = list(x_dic.keys())
while len(sub_attr) < 2:
    id = random.randint(0, len(dict_copy) - 1)
    dict = dict_copy[id]
    if dict not in sub_attr:
        sub_attr[dict] = x_dic[dict]

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
    DT = fit(train, 'EP', x_dic, labels, 1e+8, 25,2)#500
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

rf_bias = 0
rf_var = 0
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
    rf_bias += pow(y - avg, 2)
    rf_var += np.var(pred)
print(rf_bias / len(test))
print(rf_var / len(test))
