
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

test_data = df[df['Preservative_Added'].isnull()]
train_data = df[df['Preservative_Added'].notnull()]



train_data = train_data.drop(['Sample_ID','Transparent_ Window_in_Package'], axis =1)
test_data = test_data.drop(['Sample_ID','Transparent_ Window_in_Package','Preservative_Added'], axis =1)


x= train_data.drop(columns ='Preservative_Added')
#x = x.drop(columns = 'Difference_From_Fresh' )
y = train_data.loc[:,['Preservative_Added']]
x.isnull().sum().sort_values(ascending=False).head()


imp = Imputer(missing_values="NaN", strategy = 'median', axis=0)
imp.fit(x)
x1 = pd.DataFrame(data=imp.transform(x), columns=x.columns)
x1.isnull().sum().sort_values(ascending=False).head()

#x1 = x1.loc[:,['Sample_Age_weeks', 'Moidyure_%', 'Residual_oxygen_%', 'Study_Number', 'Hexanal_(ppm)', 'Processing_Agent_Stability_Index',
    #               'Storage_Conditions']]
rnd = 1
#test_size = 0.20
#Xtrain,Xtest, Ytrain, Ytest =train_test_split(x1,y, test_size=test_size, random_state = rnd)

training_model = XGBClassifier(objective = 'binary:logistic', max_depth=5, n_estimators=500)
training_model.fit(x1, y)
y_pred = training_model.predict(test_data)

#mae= mean_absolute_error(Ytest, y_pred)

#score = 1/ (1+mae)
#accuracy = training_model.score(Xtest, Ytest)
#y_pred = training_model.predict_proba(test_data)
#predict = training_model.predict(test_data)
y_predicted = training_model.predict(test_data)
#predictions = [round(value) for value in y_predicted]

#accuracy = accuracy_score(Ytest, predictions)
fig = xgb.plot_importance(training_model)

test_data['Preservative_Added']=y_predicted

#y_prediction = pd.DataFrame({'Column1': y_pred[:, 0], 'Column2': y_pred[:, 1], 'Column3': y_pred[:, 2]})


#test = y_pred.iloc[0]
result = pd.concat([train_data,test_data], join = 'outer')
result.sort_index(inplace=True)
new_data = result

export_csv = new_data.to_csv(r'C:\Users\sb00747428\Downloads\pepsico challenge\preservative_add.csv', index = None, header=True)