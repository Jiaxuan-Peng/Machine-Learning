from sklearn.metrics import accuracy_score
from ID3 import DecisionTree

if __name__ == '__main__':

    x_dic = {'buying': ['vhigh', 'high', 'med', 'low'],'maint': ['vhigh', 'high', 'med', 'low'],
             'doors': ['2', '3', '4', '5more'],'persons': ['2', '4', 'more'],
             'lug_boot': ['small', 'med', 'big'],'safety': ['low', 'med', 'high']}
    x_col = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    with open('/home/u1413911/Downloads/Machine-Learning-main/DecisionTree/car/train.csv', 'r') as f:
        for line in f:
            x_val = {}
            term = line.strip().split(',')
            for i in range(len(term) - 1):
                x_val[x_col[i]] = term[i]
            x_train.append(x_val)
            y_train.append(term[-1])

    with open('/home/u1413911/Downloads/Machine-Learning-main/DecisionTree/car/test.csv', 'r') as f:
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