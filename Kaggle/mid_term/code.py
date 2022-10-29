import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
import random
from sklearn.model_selection import GridSearchCV
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

train = pd.read_csv('train_final.csv', sep=',', header=0)
test = pd.read_csv('kaggle/test_final.csv', sep=',', header=0)

#print(train.isnull().any())
#print(test.isnull().any())

for i in train.columns:
    train[i] = np.where(train[i] == "?", train[i].mode(), train[i])
for j in test.columns:
    test[j] = np.where(test[j] == "?", random.choice(test[j].mode()), test[j])
    
y_train = train.iloc[:,-1]
x_train = train.drop("income>50K", axis=1)
x_test = test.drop("ID", axis=1)

#convert categorical value to numberic value
for i in range(x_train.shape[1]):
    k = 0
    for j in (set(x_train.iloc[:,i])):
        if (type(j)==str):
            x_train.iloc[:,i] = np.where(x_train.iloc[:,i] == j,k, x_train.iloc[:,i])
            k=k+1

for i in range(x_test.shape[1]):
    k = 0
    for j in (set(x_test.iloc[:,i])):
        if (type(j)==str):
            x_test.iloc[:,i] = np.where(x_test.iloc[:,i] == j,k, x_test.iloc[:,i])
            k=k+1


#print(x_train.columns)

s=(x_train.describe())
#s.to_csv('D:/utah/courses/y1s1/ml/kaggle/describe.csv',index=False, encoding='utf_8_sig')


#smote
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
y_train = y_train.astype('int')
print('Original dataset shape %s' % Counter(y_train))
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)
print('Resampled dataset shape %s' % Counter(y_train))


#normalization
#scaler = preprocessing.StandardScaler().fit(x_train)
scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)

pd.DataFrame(x_train,columns=['age', 'workclass', 'fnlwgt', 'education', 'education.num',
       'marital.status', 'occupation', 'relationship', 'race', 'sex',
       'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']).boxplot(grid=False, rot=45, fontsize=5)
plt.show()

#apply cross - validation for hyper - parameter selection


#report both positive and negative results




####ML method
'''
#LR
logreg = LogisticRegression(random_state=0).fit(x_train, y_train)
logreg.score(x_train, y_train)
# predict probabilities
lr_probs = logreg.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs  = lr_probs [:, 1]
lr_probs
lr_probs=pd.concat([pd.DataFrame(test["ID"],columns=['ID']), pd.DataFrame(lr_probs,columns=['Prediction'])], axis=1)
lr_probs
lr_probs.to_csv('D:/utah/courses/y1s1/ml/kaggle/LR_normalized_smote.csv',index=False, encoding='utf_8_sig')



#RF
#rf = RandomForestClassifier(n_estimators=50, max_depth = 5, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth = 10, random_state=42)
rf.fit(x_train, y_train)
rf_probs = rf.predict_proba(x_test)
rf_probs

rf_probs  = rf_probs [:, 1]
rf_probs
rf_probs=pd.concat([pd.DataFrame(test["ID"],columns=['ID']), pd.DataFrame(rf_probs,columns=['Prediction'])], axis=1)
rf_probs

rf_probs.to_csv('D:/utah/courses/y1s1/ml/kaggle/RF_normalized_smote.csv',index=False, encoding='utf_8_sig')

#adaboost
ada = AdaBoostClassifier(n_estimators=500, random_state=42)
ada.fit(x_train, y_train)
ada_probs = ada.predict_proba(x_test)
ada_probs

ada_probs  = ada_probs [:, 1]
ada_probs
ada_probs=pd.concat([pd.DataFrame(test["ID"],columns=['ID']), pd.DataFrame(ada_probs,columns=['Prediction'])], axis=1)
ada_probs

ada_probs.to_csv('D:/utah/courses/y1s1/ml/kaggle/ada_normalized_smote.csv',index=False, encoding='utf_8_sig')


#svm
svm = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale',probability=True)
svm.fit(x_train, y_train)
svm_probs = svm.predict_proba(x_test)
svm_probs

svm_probs  = svm_probs [:, 1]
svm_probs
svm_probs=pd.concat([pd.DataFrame(test["ID"],columns=['ID']), pd.DataFrame(svm_probs,columns=['Prediction'])], axis=1)
svm_probs

svm_probs.to_csv('D:/utah/courses/y1s1/ml/kaggle/svm_normalized_smote.csv',index=False, encoding='utf_8_sig')


#ann
ann = MLPClassifier(random_state=1)
ann = GridSearchCV(ann, {'hidden_layer_sizes': [10, 20, 50, 100, 200]}, verbose=1,n_jobs=2)
ann.fit(x_train, y_train)
ann_probs = ann.predict_proba(x_test)
ann_probs  = ann_probs [:, 1]
ann_probs=pd.concat([pd.DataFrame(test["ID"],columns=['ID']), pd.DataFrame(ann_probs,columns=['Prediction'])], axis=1)

ann_probs.to_csv('D:/utah/courses/y1s1/ml/kaggle/ann_normalized_smote.csv',index=False, encoding='utf_8_sig')

#xgb
xgb = xgb.XGBClassifier()
xgb = GridSearchCV(xgb, {'max_depth': [10, 15, 20],'n_estimators': [100, 200,500]}, verbose=1,n_jobs=2)
xgb.fit(x_train, y_train)
xgb_probs = xgb.predict_proba(x_test)
xgb_probs  = xgb_probs [:, 1]
xgb_probs=pd.concat([pd.DataFrame(test["ID"],columns=['ID']), pd.DataFrame(xgb_probs,columns=['Prediction'])], axis=1)
xgb_probs.to_csv('D:/utah/courses/y1s1/ml/kaggle/xgb_normalized_smote.csv',index=False, encoding='utf_8_sig')


#
gbt = GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=5)
gbt.fit(x_train, y_train)
gbt_probs = gbt.predict_proba(x_test)
gbt_probs  = gbt_probs [:, 1]
gbt_probs=pd.concat([pd.DataFrame(test["ID"],columns=['ID']), pd.DataFrame(gbt_probs,columns=['Prediction'])], axis=1)
gbt_probs.to_csv('D:/utah/courses/y1s1/ml/kaggle/gbt_normalized_smote.csv',index=False, encoding='utf_8_sig')
'''
