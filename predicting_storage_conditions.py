# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:52:47 2019

@author: sb00747428
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import sklearn.feature_selection
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df =pd.read_excel('data.xlsx')

test_data = df[df['Storage_Conditions'].isnull()]
train_data = df[df['Storage_Conditions'].notnull()]

label =[]
for row in train_data['Storage_Conditions']:
    if row == 'Warm Climate':
        label.append(0)
    elif row == 'High Temperature and Humidity':
        label.append(1)
    elif row == 'Cold Climate':
        label.append(2)

train_data['label']= label

train_data = train_data.drop(['Sample_ID','Packaging_ Stabilizer_Added','Transparent_ Window_in_Package','Preservative_Added','Storage_Conditions'], axis =1)
test_data = test_data.drop(['Sample_ID','Packaging_ Stabilizer_Added','Transparent_ Window_in_Package','Preservative_Added','Storage_Conditions'], axis =1)

x= train_data.drop(columns ='label')
#x = x.drop(columns = 'Difference_From_Fresh' )
y = train_data.loc[:,['label']]

x.isnull().sum().sort_values(ascending=False).head()


imp = Imputer(missing_values="NaN", strategy = 'median', axis=0)
imp.fit(x)
x1 = pd.DataFrame(data=imp.transform(x), columns=x.columns)
x1.isnull().sum().sort_values(ascending=False).head()

x1 = x1.loc[:,['Sample_Age_weeks', 'Moidyure_%', 'Residual_oxygen_%', 'Study_Number', 'Hexanal_(ppm)', 'Processing_Agent_Stability_Index',
                   'Storage_Conditions']]
rnd = 1
test_size = 0.10
Xtrain,Xtest, Ytrain, Ytest =train_test_split(x1,y, test_size=test_size, random_state = rnd)

training_model = XGBClassifier(objective = 'binary:logistic', max_depth=5, n_estimators=500)
training_model.fit(x1, y)
y_pred = training_model.predict(test_data)

#mae= mean_absolute_error(Ytest, y_pred)

#score = 1/ (1+mae)
#accuracy = training_model.score(Xtest, Ytest)
y_pred = training_model.predict_proba(test_data)
predict = training_model.predict(test_data)
y_predicted = training_model.predict(test_data)
predictions = [round(value) for value in y_predicted]

accuracy = accuracy_score(Ytest, predictions)
fig = xgb.plot_importance(training_model)

y_prediction = pd.DataFrame({'Column1': y_pred[:, 0], 'Column2': y_pred[:, 1], 'Column3': y_pred[:, 2]})

new_label = []
for row in y_prediction['Column1']:
    if row > 0.80 :
        new_label.append(0)
    else:
        new_label.append('nan')

y_pred['new_label']= new_label

test = y_pred.iloc[0]

if __name__ == '__main__':
    df =pd.read_excel('data.xlsx')
    todummy = ['Study_Number']
    new_data = dummy_data(df, todummy)
