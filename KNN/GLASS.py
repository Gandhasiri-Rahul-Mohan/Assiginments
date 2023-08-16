# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:02:55 2023

@author: Rahul
"""

import pandas as pd
import numpy as np

df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\KNN\\glass.csv")
df.head()
df.shape
df.duplicated()
df[df.duplicated()]
df.drop([39],axis=0,inplace=True)
df.info()

# Visualization 
import seaborn as sns
import matplotlib.pyplot as plt


# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))

# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

# Scatter plot of two features, and pairwise plot
sns.scatterplot(df['RI'],df['Na'],hue=df['Type'])


#pairwise plot of all the features
sns.pairplot(df,hue='Type')
plt.show()


# Splitting the variables
X = df.iloc[:,0:9]
X1 = df.iloc[:,1:9]
Y = df["Type"]

# Standardization
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
mm_X = MM.fit_transform(X1)

#data partition

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(mm_X,Y,test_size=0.33,random_state=(24))

# Model 1 fitting using X1


from sklearn.neighbors import KNeighborsClassifier

train_accuracy = []
test_accuracy = []

for i in range (3,24,2):
    knn = KNeighborsClassifier(n_neighbors = i, p = 2)
    knn.fit(x_train,y_train)
    y_pred_train = knn.predict(x_train)
    y_pred_test = knn.predict(x_test)

train_accuracy.append(accuracy_score(y_train,y_pred_train).round(2))
test_accuracy.append(accuracy_score(y_test,y_pred_test).round(2))
    
np.mean(train_accuracy).round(2)
np.mean(test_accuracy).round(2)


#models

models = []

from sklearn.linear_model import LogisticRegression
models.append(('LogisticRegression', LogisticRegression()))

from sklearn.svm import SVC
models.append(('SVM', SVC()))

from sklearn.tree import DecisionTreeClassifier
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))

from sklearn.ensemble import RandomForestClassifier
models.append(('Random Forest Classifier', RandomForestClassifier(max_depth=0.7)))

for title, modelname in models:
    modelname.fit(x_train, y_train)

    y_pred = modelname.predict(x_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print(title,"Accuracy: %.2f%%" % (accuracy * 100.0))

"""
LogisticRegression Accuracy: 46.48%
SVM Accuracy: 56.34%
Decision Tree Classifier Accuracy: 67.61%
Random Forest Classifier Accuracy: 21.13%

"""


# Model 2 fitting with X

# Standardization
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
mm_X = MM.fit_transform(X)

#data partition

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(mm_X,Y,test_size=0.33,random_state=(24))

from sklearn.neighbors import KNeighborsClassifier
train_accuracy_1 = []
test_accuracy_1 = []

for i in range (3,24,2):
    knn = KNeighborsClassifier(n_neighbors = i, p = 2)
    knn.fit(x_train,y_train)
    y_pred_train = knn.predict(x_train)
    y_pred_test = knn.predict(x_test)

train_accuracy_1.append(accuracy_score(y_train,y_pred_train).round(2))
test_accuracy_1.append(accuracy_score(y_test,y_pred_test).round(2))
    
np.mean(train_accuracy_1).round(2)
np.mean(test_accuracy_1).round(2)

# models
models = []

from sklearn.linear_model import LogisticRegression
models.append(('LogisticRegression', LogisticRegression()))

from sklearn.svm import SVC
models.append(('SVM', SVC()))

from sklearn.tree import DecisionTreeClassifier
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))

from sklearn.ensemble import RandomForestClassifier
models.append(('Random Forest Classifier', RandomForestClassifier(max_depth=0.7)))

for title, modelname in models:
    modelname.fit(x_train, y_train)

    y_pred = modelname.predict(x_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print(title,"Accuracy: %.2f%%" % (accuracy * 100.0))

"""

LogisticRegression Accuracy: 45.07%
SVM Accuracy: 56.34%
Decision Tree Classifier Accuracy: 66.20%
Random Forest Classifier Accuracy: 21.13%

"""














