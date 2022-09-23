import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from ID3 import DecisionTree

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

    with open('/home/u1413911/Downloads/Machine-Learning-main/DecisionTree/bank/train.csv', 'r') as f:
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

    with open('/home/u1413911/Downloads/Machine-Learning-main/DecisionTree/bank/test.csv', 'r') as f:
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
