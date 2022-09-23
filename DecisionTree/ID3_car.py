from collections import Counter
import numpy as np
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
        measure = 0
        n = len(labels)
        counter = Counter(labels)
        if self.gain == 'GI':
            measure = 1 - sum((counter[count] / n) ** 2 for count in counter)
        elif self.gain == 'ME':
            if counter.most_common(1) != []:
                measure = 1 - counter.most_common(1)[0][1]/ n
            else:
                measure = 0
        elif self.gain == 'EP':
            measure = sum(-counter[count] / n * np.log2(counter[count] / n) for count in counter)
        return measure

    def best_attr(self, S, x_dic, labels):
        max_gain = -1
        best_attr = None
        for node in x_dic:
            gain = 0
            for v in x_dic[node]:
                y_sub = [labels[j] for j in range(len(labels)) if S[j][node] == v]
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
                sub_attrs = copy.deepcopy(x_dic)
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

    x_dic = {'buying': ['vhigh', 'high', 'med', 'low'],'maint': ['vhigh', 'high', 'med', 'low'],
             'doors': ['2', '3', '4', '5more'],'persons': ['2', '4', 'more'],
             'lug_boot': ['small', 'med', 'big'],'safety': ['low', 'med', 'high']}
    x_col = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    with open('D:/utah/courses/y1s1/ml/h1/car/train.csv', 'r') as f:
        for line in f:
            x_val = {}
            term = line.strip().split(',')
            for i in range(len(term) - 1):
                x_val[x_col[i]] = term[i]
            x_train.append(x_val)
            y_train.append(term[-1])

    with open('D:/utah/courses/y1s1/ml/h1/car/test.csv', 'r') as f:
        for line in f:
            x_val = {}
            term = line.strip().split(',')
            for i in range(len(term) - 1):
                x_val[x_col[i]] = term[i]
            x_test.append(x_val)
            y_test.append(term[-1])

    for j in range(1,7):
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
