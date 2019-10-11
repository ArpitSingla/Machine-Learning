# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 01:38:20 2019

@author: Arpit Singla
"""

import numpy as mp
import pandas as pd

dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',max_depth=3)
classifier=classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn import metrics
a=metrics.accuracy_score(y_pred,y_test)
print(a)
