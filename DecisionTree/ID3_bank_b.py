from collections import Counter
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import accuracy_score


class Tree:
    def __init__(self, node):
        self.label = None
        self.node = node
        self.child = {}


class DecisionTree():

    def __init__(self, gain, max_dep, labels, x_dic):

        self.gain = gain
        self.max_dep = max_dep
        self.labels = labels
        self.x_dic = x_dic

    def cal_gain(self, labels):
        total_gain = 0
        n = len(labels)
        count = Counter(labels)
        if self.gain == 'GI':
            total_gain = 1 - sum((count[i]/n) ** 2 for i in count)
        elif self.gain == 'ME':
            if max(count.values(), default=0) != 0:
                total_gain = 1 - max(count.values())/ n
            else:
                total_gain = 0
        elif self.gain == 'EP':
            total_gain = sum(-count[i]/n * np.log2(count[i]/n) for i in count)
        return total_gain

    def best_attr(self, S, x_dic, labels):
        max_gain = -1
        for node in x_dic:
            gain = 0
            for v in x_dic[node]:
                y_sub = []
                for j in range(len(labels)):
                    if S[j][node] == v:
                        y_sub.append(labels[j])
                gain += (len(y_sub) / len(labels)) * self.cal_gain(y_sub)
            if (self.cal_gain(labels) - gain) > max_gain:
                max_gain = self.cal_gain(labels) - gain
                best_attr = node
        return best_attr

    def ID3(self, S, x_dic, labels, max_dep):
        if len(set(labels)) == 1 or len(x_dic) == 0 or max_dep == 0:
            leaf = Tree(None)
            leaf.label = Counter(labels).most_common(1)[0][0]
            return leaf

        A = self.best_attr(S, x_dic, labels)
        root = Tree(A)
        for v in x_dic[A]:
            S_val = []
            S_val_labels = []
            new_branch = Tree(v)
            for i in range(len(S)):
                if S[i][A] == v:
                    S_val.append(S[i])
                    S_val_labels.append(labels[i])
            if S_val ==[]:
                new_branch.label = Counter(labels).most_common(1)[0][0]
                root.child[v]=new_branch
            else:
                sub_attrs = copy.copy(x_dic)
                sub_attrs.pop(A)
                root.child[v] = self.ID3(S_val, sub_attrs, S_val_labels, max_dep - 1)
        return root

    def ID3_rev(self, S):
        self.root = self.ID3(S, self.x_dic, self.labels, self.max_dep)
    def fit(self, S):
        y_pred = []
        for x_val in S:
            root = self.root
            while root.child:
                node = x_val[root.node]
                if node in root.child:
                    root = root.child[node]
            y_pred.append(root.label)
        return y_pred


if __name__ == '__main__':

    x_dic = {'age': [0, 1],'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
        'marital': ['married','divorced','single'],'education': ['unknown', 'secondary', 'primary', 'tertiary'],
        'default': ['yes', 'no'],'balance': [0, 1],'housing': ['yes', 'no'],'loan': ['yes', 'no'],
        'contact': ['unknown', 'telephone', 'cellular'],'day': [0, 1],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'duration': [0, 1], 'campaign': [0, 1], 'pdays': [0, 1], 'previous': [0, 1], 'poutcome': ['unknown', 'other', 'failure', 'success']}
    x_dic_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    x_dic_unknown=['job', 'education', 'contact', 'poutcome']

    x_col = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays','previous' , 'poutcome']

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    with open('D:/utah/courses/y1s1/ml/h1/bank/train.csv', 'r') as f:
        for line in f:
            x_val = {}
            term = line.strip().split(',')
            for i in range(len(term) - 1):
                x_val[x_col[i]] = term[i]
            x_train.append(x_val)
            y_train.append(term[-1])

    #use the median to convert the value
    x_traindf = pd.DataFrame(x_train)
    for i in x_dic_numeric:
        x_traindf[i] = np.where(x_traindf[i].astype(int) <=(int(x_traindf[i].median())), 0, x_traindf[i])
        x_traindf[i] = np.where(x_traindf[i].astype(int) >(int(x_traindf[i].median())), 1, x_traindf[i])
    for i in x_dic_unknown:
        if (x_traindf[i].mode()== "unknown").bool():
            x_traindf[i] = np.where(x_traindf[i]=="unknown", x_traindf[i].value_counts().index.tolist()[1], x_traindf[i])
        else:
            x_traindf[i] = np.where(x_traindf[i]=="unknown", x_traindf[i].mode(), x_traindf[i])
    x_train = x_traindf.to_dict('records')

    with open('D:/utah/courses/y1s1/ml/h1/bank/test.csv', 'r') as f:
        for line in f:
            x_val = {}
            term = line.strip().split(',')
            for i in range(len(term) - 1):
                x_val[x_col[i]] = term[i]
            x_test.append(x_val)
            y_test.append(term[-1])

    #use the median to convert the value
    x_testdf = pd.DataFrame(x_test)
    for i in x_dic_numeric:
        x_testdf[i] = np.where(x_testdf[i].astype(int) <=(int(x_testdf[i].median())), 0, x_testdf[i])
        x_testdf[i] = np.where(x_testdf[i].astype(int) >(int(x_testdf[i].median())), 1, x_testdf[i])
    x_test = x_testdf.to_dict('records')
    for i in x_dic_unknown:
        if (x_testdf[i].mode()== "unknown").bool():
            x_testdf[i] = np.where(x_testdf[i]=="unknown", x_testdf[i].value_counts().index.tolist()[1], x_testdf[i])
        else:
            x_testdf[i] = np.where(x_testdf[i]=="unknown", x_testdf[i].mode(), x_testdf[i])
    x_test = x_testdf.to_dict('records')

    for j in range(1,17):
        for i in ["EP", "ME","GI"]:
            error_train = []
            error_test = []
            cls = DecisionTree(gain=i, max_dep=j,labels=y_train,x_dic=x_dic)
            cls.ID3_rev(x_train)
            y_fit = cls.fit(x_train)
            error_train.append(1 - accuracy_score(y_train,y_fit))
            y_pred = cls.fit(x_test)
            error_test.append(1 - accuracy_score(y_test, y_pred))
            print(i,j,error_train,error_test)
