# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 23:10:08 2019

@author: sb00747428
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
# dataset
from sklearn.datasets import load_boston
# scaling and dataset split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# OLS, Ridge
from sklearn.linear_model import LinearRegression, Ridge
# model evaluation
from sklearn.metrics import r2_score, mean_squared_error

# load dataset
df =pd.read_excel('data1.xlsx')
def dummy_data(df, todummy):
    for x in todummy:
        dummies = pd.get_dummies(df[x], prefix = x, dummy_na = False)
        df= df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df
todummy = ['Study_Number']
new_data = dummy_data(df, todummy)

df_miss = new_data.drop(['Sample_ID','Transparent_ Window_in_Package','Sample_Age_(Weeks)'], axis =1)
df_miss1 = new_data.drop(['Sample_ID','Transparent_ Window_in_Package','Sample_Age_(Weeks)'], axis =1)
df_final = new_data.drop(['Sample_ID','Transparent_ Window_in_Package'], axis =1)

#for x in todummy:
#    dummies = pd.get_dummies(df[x], prefix = x, dummy_na = False)
#    df= df.drop(x, 1)
#    df = pd.concat([df, dummies], axis=1)
#new_data

# train the model

maxdiff = df_miss['Difference_From_Fresh'].max()
df_miss['Difference_From_Fresh']= df_miss['Difference_From_Fresh']/maxdiff

maxmos = df_miss['Moisture_(%)'].max()
df_miss['Moisture_(%)']= df_miss['Moisture_(%)']/maxmos

maxoxy = df_miss['Residual_Oxygen_(%)'].max()
df_miss['Residual_Oxygen_(%)']= df_miss['Residual_Oxygen_(%)']/maxoxy

maxhex = df_miss['Hexanal_(ppm)'].max()
df_miss['Residual_Oxygen (%)']= df_miss['Hexanal_(ppm)']/maxhex

x=pd.DataFrame(columns=['Moisture_(%)'])
imputdf = df_miss.drop(['Moisture_(%)'], axis =1)
x['Moisture_(%)'] = df_miss['Moisture_(%)']

imputdf.isnull().sum().sort_values(ascending=False).head()
imp = Imputer(missing_values="NaN", strategy = 'median', axis=0)
imp.fit(imputdf)
imputdf = pd.DataFrame(data=imp.transform(imputdf), columns=imputdf.columns)
imputdf['Moisture_(%)']=x['Moisture_(%)']

testdf = imputdf[df_miss['Moisture_(%)'].isnull()]
traindf = imputdf[df_miss['Moisture_(%)'].notnull()]
trainX = traindf.drop(['Moisture_(%)'], axis =1)
testX = testdf.drop(['Moisture_(%)'], axis =1)
trainY = traindf.loc[:,['Moisture_(%)']]
# standardize and train/test split
house_price.data = preprocessing.scale(house_price.data)
X_train, X_test, y_train, y_test = train_test_split(
    trainX,trainY, test_size=0.1, random_state=1)

# initialize
ridge_reg = Ridge(alpha=0)
ridge_reg.fit(X_train, y_train)
ridge_df = pd.DataFrame({'variable': house_price.feature_names, 'estimate': ridge_reg.coef_})
ridge_train_pred = []
ridge_test_pred = []

# iterate lambdas
for alpha in np.arange(0, 200, 1):
    # training
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X_train, y_train)
    var_name = 'estimate' + str(alpha)
    ridge_df[var_name] = ridge_reg.coef_
    # prediction
    ridge_train_pred.append(ridge_reg.predict(X_train))
    ridge_test_pred.append(ridge_reg.predict(X_test))

# organize dataframe
ridge_df = ridge_df.set_index('variable').T.rename_axis('estimate').rename_axis(None, 1).reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ridge_df.RM, 'r', ridge_df.ZN, 'g', ridge_df.RAD, 'b', ridge_df.CRIM, 'c', ridge_df.TAX, 'y')
ax.axhline(y=0, color='black', linestyle='--')
ax.set_xlabel("Lambda")
ax.set_ylabel("Beta Estimate")
ax.set_title("Ridge Regression Trace", fontsize=16)
ax.legend(labels=['Room','Residential Zone','Highway Access','Crime Rate','Tax'])
ax.grid(True)
ridge_mse_test = [mean_squared_error(y_test, p) for p in ridge_test_pred]
ols_mse = mean_squared_error(y_test, ols_pred)

# plot mse
plt.plot(ridge_mse_test[:25], 'ro')
plt.axhline(y=ols_mse, color='g', linestyle='--')
plt.title("Ridge Test Set MSE", fontsize=16)
plt.xlabel("Model Simplicity$\longrightarrow$")
plt.ylabel("MSE")
