import math
import copy
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pandas as pd


class Tree:
    def __init__(self, node):
        self.node = node
        self.child = {}
    def Child(self):
        return (len(self.child) == 0)

def cal_gain(dataset, gain, x_dic, x_v):
    total_gain_temp = 0
    total = 0
    g = 0
    for i in dataset:
        total += i['w']
    c = {}
    for i in dataset:
        label = i['label']
        if label not in c:
            c[label] = 0
        c[label] += i['w']
        if gain == 'EP':
            for (label, count) in c.items():
                g += -count / total * np.log2(count / total)
        elif gain == 'GI':
            temp = 0
            for (label, count) in c.items():
                temp += count / total ** 2
                g = 1 - temp
        elif gain == 'ME':
            temp = 0
            for (label, count) in c.items():
                g = 1 - max(temp, count / total)
    for v in x_v:
        sub_set = []
        for i in dataset:
            if i[x_dic] == v:
                sub_set.append(i)
        sub_total = 0
        for i in sub_set:
            sub_total += i['w']
        total_gain_temp += sub_total / total * g
        total_gain = g - total_gain_temp
    return total_gain

def Node(i, dt):
    update_dt = dt
    while not update_dt.Child():
        attr = update_dt.node
        attr_v = i[attr]
        update_dt = update_dt.child[attr_v]
    return update_dt.node

def ID3(dataset, gain, x_dic, labels, max_dep):
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
            sub_x_dic = copy.deepcopy(x_dic)
            sub_x_dic.pop(best_attr)
            sub_labels = set()
            for i in sub_set:
                sub_label = i['label']
                if sub_labels not in sub_labels:
                    sub_labels.add(sub_label)
            root.child[v] = ID3(sub_set, gain, sub_x_dic, sub_labels, max_dep-1)
    return root

def pred(dataset, DT):
    r = 0
    for i in dataset:
        pred = 0
        for dt in DT:
            node = Node(i, dt)
            node = 1 if node == 'yes' else -1
            pred += node
        if i['label'] == 'yes' and pred > 0:
            r += 1
        if i['label'] == 'no' and pred < 0:
            r += 1
    return r / len(dataset)