# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:24:43 2019

@author: sb00747428
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
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










if __name__ == '__main__':
    df =pd.read_excel('data.xlsx')
    todummy = ['Study_Number']
    new_data = dummy_data(df, todummy)

    df_miss = new_data.drop(['Sample_ID','Transparent_ Window_in_Package','Sample_Age_(Weeks)'], axis =1)
    df_final = new_data.drop(['Sample_ID','Transparent_ Window_in_Package'], axis =1)


    
