# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:28:32 2021

@author: leeso
"""

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

##################################################################################
##################################################################################


os.chdir("C:\\Users\\leeso\\Downloads\\신용카드")

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
submission=pd.read_csv('sample_submission.csv')
data=pd.concat([train, test], axis=0)

data.isnull().sum()
data = data.drop('occyp_type', axis=1)

data.info()


for i in data.columns :
    print(data[i].value_counts())
    print('---------------------')
    
'''
columns = ['index', 'gender', 'car', 'reality', **'child_num', **'income_total',
       'income_type', 'edu_type', 'family_type', 'house_type', **'DAYS_BIRTH',
       **'DAYS_EMPLOYED', 'FLAG_MOBIL', 'work_phone', 'phone', 'email',
       **'family_size', **'begin_month', **'credit']
'''
data['edu_type'].value_counts()
def data_transform(data) :
    label_encoder=preprocessing.LabelEncoder()
    data = data.drop('index',axis=1)
    
    data['gender'] = data['gender'].replace(['F','M'],[0,1])
    data['car'] = data['car'].replace(['N','Y'],[0,1])
    data['reality'] = data['reality'].replace(['N','Y'],[0,1])
    data['edu_type'] = data['edu_type'].replace(['Lower secondary','Secondary / secondary special','Incomplete higher','Higher education','Academic degree'],[0,1,2,3,4])
    data['income_total'] = data['income_total']/10000

    #data['income_type']=label_encoder.fit_transform(data['income_type'])
    
    enc = OneHotEncoder(drop='first')

    object_cols= ['income_type','family_type','house_type']
    enc.fit(data.loc[:,object_cols])
    train_onehot_df = pd.DataFrame(enc.transform(data.loc[:,object_cols]).toarray(),columns=enc.get_feature_names(object_cols))
    data.drop(object_cols, axis=1, inplace=True)
    train_onehot_df = train_onehot_df.reset_index(drop=True)
    data = data.reset_index(drop=True)
    data = pd.concat([train_onehot_df, data], axis=1)  
    
    #data['family_type']=label_encoder.fit_transform(data['family_type'])
    #data['house_type']=label_encoder.fit_transform(data['house_type'])

    return(data)
    
data = data_transform(data)

'''
columns = ['index', 'gender', 'car', 'reality', 'child_num', 'income_total',
       'income_type', 'edu_type', 'family_type', 'house_type', **'DAYS_BIRTH',
       **'DAYS_EMPLOYED', 'FLAG_MOBIL', 'work_phone', 'phone', 'email',
       'family_size', **'begin_month', **'credit']
'''

label_encoder=preprocessing.LabelEncoder()

data['child_num'].value_counts()
data['child_num'].value_counts(sort=False).plot.bar()
data.loc[data['child_num'] >= 3,'child_num']=3

data['income_total']
data['income_total'].plot(kind='hist',bins=10,density=True) # 범주화
count, bin_dividers =np.histogram(data['income_total'], bins=10)
bin_names=['income'+str(i) for i in range(10) ]
data['income_total']=pd.cut(x=data['income_total'], bins=bin_dividers, labels=bin_names, include_lowest=True)
data['income_total']=label_encoder.fit_transform(data['income_total'])

data['family_size'].value_counts()
data['family_size'].value_counts(sort=True).plot.bar()
data.loc[data['family_size'] >= 6,'family_size']=6


variable = ['DAYS_BIRTH','DAYS_EMPLOYED','begin_month']
data[variable]=-data[variable]


'''
def make_bin(variable, n):
    data[variable]=-data[variable]
#    count, bin_dividers =np.histogram(data[variable], bins=n)
    bin_names=[str(i) for i in range(n)]
    data[variable]=pd.cut(x=data[variable], bins=bin_dividers, labels=bin_names, include_lowest=True)
    data[variable]=label_encoder.fit_transform(data[variable])
'''

data['DAYS_BIRTH'].plot(kind='hist',bins=30,density=True)
data['DAYS_EMPLOYED'].plot(kind='hist',bins=30,density=True)
data['begin_month'].plot(kind='hist',bins=30,density=True)
'''
make_bin('DAYS_BIRTH', n=10)
make_bin('DAYS_EMPLOYED', n=6)
make_bin('begin_month', n=4)
'''
data.isnull().sum()
data.info()
#data = data.drop('index',axis=1)


##################################################################################
##################################################################################

train=data[:len(data)-10000]
test=data[len(data)-10000:]

train_x=train.drop('credit', axis=1)
train_y=train[['credit']]
test_x=test.drop('credit', axis=1)

print(train_x.shape, train_y.shape, test_x.shape)


X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, 
                                                    stratify=train_y, test_size=0.25,
                                                    random_state = 10086)


lgb = LGBMClassifier(n_estimators=1000, num_leaves=50, subsample=0.8, min_child_samples=60, max_depth=20)
evals = [(X_val, y_val)]
lgb.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='logloss',
         eval_set=evals, verbose=True)
y_pred = lgb.predict_proba(X_val)

from tensorflow.keras.utils import to_categorical

print(f"log_loss: {log_loss(to_categorical(y_val['credit']), y_pred)}")


from sklearn.model_selection import KFold, StratifiedKFold

def run_kfold(clf):
    folds=StratifiedKFold(n_splits=5, shuffle=True, random_state=55)
    outcomes=[]
    sub=np.zeros((test_x.shape[0], 3))  
    for n_fold, (train_index, val_index) in enumerate(folds.split(train_x, train_y)):
        X_train, X_val = train_x.iloc[train_index], train_x.iloc[val_index]
        y_train, y_val = train_y.iloc[train_index], train_y.iloc[val_index]
        lgb.fit(X_train, y_train)
        
        predictions=clf.predict_proba(X_val)
        
        logloss=log_loss(to_categorical(y_val['credit']), predictions)
        outcomes.append(logloss)
        print(f"FOLD {n_fold} : logloss:{logloss}")
        
        sub+=lgb.predict_proba(test_x)
        
        
    mean_outcome=np.mean(outcomes)
    
    print("Mean:{}".format(mean_outcome))
    return sub/folds.n_splits

my_submission = run_kfold(lgb)
my_submission
submission.loc[:,1:]=my_submission
submission
submission.to_csv('lgbm_submission_0420.csv', index=False)
