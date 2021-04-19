# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 18:38:57 2021

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



os.chdir("C:\\Users\\leeso\\Downloads\\신용카드")

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
submission=pd.read_csv('sample_submission.csv')

data=pd.concat([train, test], axis=0)

data.isnull().sum()
data = data.drop('occyp_type', axis=1)

data.info()


def data_cleansing(data) :
    data['gender'] = data['gender'].replace(['F','M'],[0,1])
    data['car'] = data['car'].replace(['N','Y'],[0,1])
    data['reality'] = data['reality'].replace(['N','Y'],[0,1])
    
    return(data)
    
data = data_cleansing(data)

print(data['gender'].value_counts())
print(data['car'].value_counts())
print(data['reality'].value_counts())
print(data['child_num'].value_counts())
print(data['income_type'].value_counts())
print(data['edu_type'].value_counts())
print(data['family_type'].value_counts())
print(data['house_type'].value_counts())
print(data['FLAG_MOBIL'].value_counts())
print(data['work_phone'].value_counts())
print(data['phone'].value_counts())
print(data['email'].value_counts())
print(data['family_size'].value_counts())
print(data['credit'].value_counts())


data['child_num'].value_counts(sort=False).plot.bar()
data.loc[data['child_num'] >= 3,'child_num']=3

data['income_total'] 
data['income_total'] = data['income_total']/10000

data['income_total'].plot(kind='hist',bins=10,density=True)
count, bin_dividers =np.histogram(data['income_total'], bins=10)

bin_names=['income'+str(i) for i in range(10) ]
data['income_total']=pd.cut(x=data['income_total'], bins=bin_dividers, labels=bin_names, include_lowest=True)

print(data['income_type'].unique())
print(data['edu_type'].unique())
print(data['family_type'].unique())
print(data['house_type'].unique())

data['income_total']
label_encoder=preprocessing.LabelEncoder()
data['income_type']=label_encoder.fit_transform(data['income_type'])
data['edu_type']=label_encoder.fit_transform(data['edu_type'])
data['family_type']=label_encoder.fit_transform(data['family_type'])
data['house_type']=label_encoder.fit_transform(data['house_type'])
data['income_total']=label_encoder.fit_transform(data['income_total'])



def make_bin(variable, n):
    data[variable]=-data[variable]
    count, bin_dividers =np.histogram(data[variable], bins=n)
    bin_names=[str(i) for i in range(n)]
    data[variable]=pd.cut(x=data[variable], bins=bin_dividers, labels=bin_names, include_lowest=True)
    data[variable]=label_encoder.fit_transform(data[variable])


data['DAYS_BIRTH'].plot(kind='hist',bins=10,density=True)
data['DAYS_EMPLOYED'].plot(kind='hist',bins=6,density=True)
data['begin_month'].plot(kind='hist',bins=4,density=True)

make_bin('DAYS_BIRTH', n=10)
make_bin('DAYS_EMPLOYED', n=6)
make_bin('begin_month', n=4)

data.isnull().sum()
data.info()

train=data[:len(data)-10000]
test=data[len(data)-10000:]

