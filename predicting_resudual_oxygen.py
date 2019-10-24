# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:47:50 2019

@author: sb00747428
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:24:43 2019

@author: sb00747428
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

def dummy_data(df, todummy):
    for x in todummy:
        dummies = pd.get_dummies(df[x], prefix = x, dummy_na = False)
        df= df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(100, input_dim=dim, activation="relu"))
	#model.add(Dense(4, activation="relu"))

	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))

	# return our model
	return model

def predict_Residual_Oxygen(df_miss):
    x=pd.DataFrame(columns=['Residual_Oxygen_(%)'])
    imputdf = df_miss.drop(['Residual_Oxygen_(%)'], axis =1)
    x['Residual_Oxygen_(%)'] = df_miss['Residual_Oxygen_(%)']

    imputdf.isnull().sum().sort_values(ascending=False).head()
    imp = Imputer(missing_values="NaN", strategy = 'median', axis=0)
    imp.fit(imputdf)
    imputdf = pd.DataFrame(data=imp.transform(imputdf), columns=imputdf.columns)
    imputdf['Residual_Oxygen_(%)']=x['Residual_Oxygen_(%)']

    testdf = imputdf[df_miss['Residual_Oxygen_(%)'].isnull()]
    traindf = imputdf[df_miss['Residual_Oxygen_(%)'].notnull()]
    trainX = traindf.drop(['Residual_Oxygen_(%)'], axis =1)
    testX = testdf.drop(['Residual_Oxygen_(%)'], axis =1)
    trainY = traindf.loc[:,['Residual_Oxygen_(%)']]
    #testY =testdf.loc[:,['Moisture_(%)']]
    #trainX.isnull().sum().sort_values(ascending=False).head()

    return trainX, testX, trainY, traindf, testdf




if __name__ == '__main__':
    df =pd.read_excel('data.xlsx')
    todummy = ['Study_Number']
    new_data = dummy_data(df, todummy)

    df_miss = new_data.drop(['Sample_ID','Transparent_ Window_in_Package','Sample_Age_(Weeks)'], axis =1)
    df_miss1 = new_data.drop(['Sample_ID','Transparent_ Window_in_Package','Sample_Age_(Weeks)'], axis =1)
    df_final = new_data.drop(['Sample_ID','Transparent_ Window_in_Package'], axis =1)



    # train the model

    maxdiff = df_miss['Difference_From_Fresh'].max()
    df_miss['Difference_From_Fresh']= df_miss['Difference_From_Fresh']/maxdiff

    maxmos = df_miss['Moisture_(%)'].max()
    df_miss['Moisture_(%)']= df_miss['Moisture_(%)']/maxmos

    maxoxy = df_miss['Residual_Oxygen_(%)'].max()
    df_miss['Residual_Oxygen_(%)']= df_miss['Residual_Oxygen_(%)']/maxoxy

    maxhex = df_miss['Hexanal_(ppm)'].max()
    df_miss['Hexanal_(ppm)']= df_miss['Hexanal_(ppm)']/maxhex


    trainX, testX, trainY, traindf, testdf = predict_Residual_Oxygen(df_miss)
    '''validation check'''
#    Xtrain,Xtest, Ytrain, Ytest =train_test_split(trainX,trainY, test_size=0.10, random_state = 1)
#    model = create_mlp(trainX.shape[1], regress=True)
#    opt = Adam(lr=1e-4, decay=1e-3 / 51)
#    model.compile(loss="mean_squared_error", optimizer=opt)
#    model.fit(Xtrain, Ytrain,validation_data=(Xtest, Ytest),epochs=50, batch_size=10)
#    preds = model.predict(Xtest)
#    diff = preds - Ytest
#    percentDiff = (diff / Ytest) * 100
#    absPercentDiff = np.abs(percentDiff)
#    mean = np.mean(absPercentDiff)
#    std = np.std(absPercentDiff)
    '''                                      '''
    print("[INFO] training model...")
    #testY = test["price"] / maxPrice
    model = create_mlp(trainX.shape[1], regress=True)
    opt = Adam(lr=1e-4, decay=1e-3 / 50)
    model.compile(loss="mean_squared_error", optimizer=opt)
    model.fit(trainX, trainY,epochs=50, batch_size=10)
    preds = model.predict(testX)
    testX['Residual_Oxygen_(%)']=preds

    result = pd.concat([traindf,testX], join = 'outer')
    result.sort_index(inplace=True)
    result['Residual_Oxygen_(%)'] = result['Residual_Oxygen_(%)']*maxmos
    df_miss1['Residual_Oxygen_(%)'] = result['Residual_Oxygen_(%)']
    export_csv = df_miss1.to_csv (r'C:\Users\sb00747428\Downloads\pepsico challenge\predict_Residual_Oxygen.csv', index = None, header=True)
