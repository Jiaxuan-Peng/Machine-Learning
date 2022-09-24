from collections import Counter
import numpy as np
import copy


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
        total_gain_temp = 0
        n = len(labels)
        count = Counter(labels)
        if self.gain == 'GI':
            for i in count:
                total_gain_temp += (count[i]/n) ** 2
                total_gain = 1 - total_gain_temp
        elif self.gain == 'ME':
            if max(count.values(), default=0) != 0:
                total_gain = 1 - max(count.values())/ n
            else:
                total_gain = 0
        elif self.gain == 'EP':
            for i in count:
                total_gain += -count[i]/n * np.log2(count[i]/n)
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
        if len(np.unique(labels)) == 1 or len(x_dic) == 0 or max_dep == 0:
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
