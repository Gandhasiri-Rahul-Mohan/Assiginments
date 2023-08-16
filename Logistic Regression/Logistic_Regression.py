# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 10:14:24 2022

@author: Rahul
"""

import pandas as pd
import numpy as np
df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Logistic Regression\\bank-full.csv", sep=';')
df
df.head()
df.isnull().sum() # There is no Null values

df.dtypes
df.shape

df.columns
df.duplicated()
df[df.duplicated()]

df.info()

# Evaluating Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns
# Histogram
df["age"].hist()
df["age"].skew()

df["balance"].hist()
df["balance"].skew()

df["day"].hist()
df["day"].skew()

df["duration"].hist()
df["duration"].skew()

df["campaign"].hist()
df["campaign"].skew()

df["pdays"].hist()
df["pdays"].skew()

df["previous"].hist()
df["previous"].skew()

df.boxplot("age",vert=False)
Q1=np.percentile(df["age"],25)
Q3=np.percentile(df["age"],75)
IQR=Q3-Q1
LW=Q1-(2.5*IQR)
UW=Q3+(2.5*IQR)
df["age"]<LW
df[df["age"]<LW].shape
df["age"]>UW
df[df["age"]>UW].shape
df["age"]=np.where(df["age"]>UW,UW,np.where(df["age"]<LW,LW,df["age"]))

df.boxplot("balance",vert=False)
Q1=np.percentile(df["balance"],25)
Q3=np.percentile(df["balance"],75)
IQR=Q3-Q1
LW=Q1-(2.5*IQR)
UW=Q3+(2.5*IQR)
df["balance"]<LW
df[df["balance"]<LW].shape
df["balance"]>UW
df[df["balance"]>UW].shape
df["balance"]=np.where(df["age"]>UW,UW,np.where(df["age"]<LW,LW,df["age"]))

df.boxplot("day",vert=False)

df.boxplot("duration",vert=False)
Q1=np.percentile(df["duration"],25)
Q3=np.percentile(df["duration"],75)
IQR=Q3-Q1
LW=Q1-(2.5*IQR)
UW=Q3+(2.5*IQR)
df["duration"]<LW
df[df["duration"]<LW].shape
df["duration"]>UW
df[df["duration"]>UW].shape
df["duration"]=np.where(df["duration"]>UW,UW,np.where(df["duration"]
                                                      <LW,LW,df["duration"]))

df.boxplot("campaign",vert=False)
Q1=np.percentile(df["campaign"],25)
Q3=np.percentile(df["campaign"],75)
IQR=Q3-Q1
LW=Q1-(2.5*IQR)
UW=Q3+(2.5*IQR)
df["campaign"]<LW
df[df["campaign"]<LW].shape
df["campaign"]>UW
df[df["campaign"]>UW].shape
df["campaign"]=np.where(df["campaign"]>UW,UW,np.where(df["campaign"]
                                                      <LW,LW,df["campaign"]))

plt.scatter(x=df["age"],y=df["balance"])

plt.scatter(x=df["day"],y=df["duration"])

plt.scatter(x=df["campaign"],y=df["pdays"])

plt.scatter(x=df["previous"],y=df["balance"])

#spliting into contineous and categorical

X1 = df[df.columns[[0,5,9,11,12,13,14]]]
X1

X2 = df[df.columns[[1,2,3,4,6,7,8,10,15,16]]]
X2

# Data Transformation 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for col in X2:
    # Create Object of LabelEncoder
    LE = LabelEncoder()
    X2[col] = LE.fit_transform(X2[col])
X2.dtypes

X = pd.concat([X1,X2], axis = 1)
X.head()

# Split dataset in input and output
x = X.drop('y',axis=1)  # input
y1 = X['y']             # output
x.head()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_x = ss.fit_transform(x)

# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y1,test_size=0.3)

# Model Fitting
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, Y_train)
Y_pred_train = LR.predict(X_train)
Y_pred_test = LR.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, log_loss
ac_1 = accuracy_score(Y_train, Y_pred_train)
ac_2 = accuracy_score(Y_test, Y_pred_test)
print("accuracy_score_1: ", ac_1)
print("accuracy_score_2: ", ac_2)


fs_1 = f1_score(Y_train, Y_pred_train)
fs_2 = f1_score(Y_test, Y_pred_test)
print("f1_score_1: ", fs_1)
print("f1_score_2: ", fs_2)

ll_1 = log_loss(Y_train, Y_pred_train)
ll_2 = log_loss(Y_test, Y_pred_test)
print("Log_Loss_1: ", ll_1)
print("Log_Loss_2: ", ll_2)























