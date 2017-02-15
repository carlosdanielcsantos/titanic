#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:57:05 2017

@author: carlos
"""

import numpy as np
import pandas
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

class Predictor:
    def __init__(self, predictor, data, features, labels, categorical=None, \
                 poly_deg=None):
        
        # ignore nans only from relevant features
        prev_len = len(data)
        data = data.dropna(subset=features+[labels])
        
        # get np array of features
        self.X = np.array(data[features])
        
        self.poly_deg = poly_deg
        self.categorical = categorical
        
        if self.categorical is not None:
            # anonym function to find indexes of categorica features in X
            find_categorical = lambda features, categ: \
            [features.index(e) for e in categ]
         
            # encode categorical features into one hot encoding
            self.encoder = OneHotEncoder(categorical_features = \
                            find_categorical(features, self.categorical)) 
            self.encoder.fit(self.X)
            self.X = self.encoder.transform(self.X).toarray()
        
        if self.poly_deg is not None:
            self.poly_deg = PolynomialFeatures(degree=poly_deg)
            self.poly_deg.fit(self.X)
            self.X = self.poly_deg.transform(self.X)

        self.scaler = MaxAbsScaler()
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)
        
        # get np array of labels
        self.y = np.array(data[labels])
    
        self.predictor = predictor
        self.features = features
        
        print "*** Predictor for", labels, " ***"
        print "Features =", features, "(categorical: ", categorical, ")"
        print "Data size = ", len(data), "/", prev_len
        
        self.Train(0.8)
        
    def Predict(self, data):
        if data.empty:
            return []
        # ignore nans
        data = data.dropna(subset=self.features)
        
        # get np array of features
        X = np.array(data[self.features])
        
        # transform categorical data
        if self.categorical is not None:
            # encode categorical features
            X = self.encoder.transform(X).toarray()
        
        # create polynomial features
        if self.poly_deg is not None:
            X = self.poly_deg.transform(X)
            
        # scale new data with scaler
        X = self.scaler.transform(X)
        
        return self.predictor.predict(X)
    
    def Train(self, train_ratio):
        X_train, X_test, y_train, y_test = train_test_split(self.X, \
                            self.y,test_size=1-train_ratio, random_state=42)
        
        self.predictor.fit(X_train, y_train)
        
        print "Training/Test set score = ", \
            self.predictor.score(X_train, y_train), " / ", \
            self.predictor.score(X_test, y_test), "\n"
        #mean_absolute_error(y_train, self.predictor.predict(X_train))

        #mean_absolute_error(y_test, self.predictor.predict(X_test))
        
def transformFeatureIntoNumbers(data, feature):
    for i in range(len(data[feature].dropna().unique())):
        data[feature].replace(data[feature].dropna().unique()[i], i, \
                          inplace=True)
    
test = pandas.read_csv("test.csv")
train = pandas.read_csv("train.csv")

# merge datasets to pre-process data
data = pandas.concat([test,train])


# create new column with Name length
data["Namel"] = data.Name.str.len()
train["Namel"] = train.Name.str.len()
test["Namel"] = test.Name.str.len()

# drop unuseful features
data.drop(["Name", "Ticket"],axis=1, inplace=True)
test.drop(["Name", "Ticket"],axis=1, inplace=True)
train.drop(["Name", "Ticket"],axis=1, inplace=True)

# extract just the letter from cabin
for c in 'ABCDEFG':
    data.loc[data.Cabin.str.contains(c, regex=True, na=False), "Cabin"] = c


# transform string features into numbers
transformFeatureIntoNumbers(data, "Sex")
transformFeatureIntoNumbers(train, "Sex")
transformFeatureIntoNumbers(test, "Sex")
transformFeatureIntoNumbers(data, "Embarked")
transformFeatureIntoNumbers(train, "Embarked")
transformFeatureIntoNumbers(test, "Embarked")
transformFeatureIntoNumbers(data, "Cabin")
transformFeatureIntoNumbers(train, "Cabin")
transformFeatureIntoNumbers(test, "Cabin")

# count nans in each column, drop columns with more than 50% nans
for k in data.keys():
    nanRatio = 100.0*(len(data)-data[k].count())/len(data)
    print k, nanRatio, '%', "(", len(data)-data[k].count(),")"
    if (nanRatio > 90):
        data.drop(k, axis=1, inplace=True)
        test.drop(k, axis=1, inplace=True)
        train.drop(k, axis=1, inplace=True)
        print "\tDropped", k


age = Predictor(Ridge(alpha=1), data, \
        list(data.keys().difference(["Age","Cabin","Survived","PassengerId"])), \
        "Age", ["Embarked"], poly_deg=5)

fare = Predictor(Ridge(alpha=1), data, \
        list(data.keys().difference(["Fare","Cabin","Survived","PassengerId"])), \
       "Fare", ["Embarked"], poly_deg=5)

embarked = Predictor(LogisticRegression(), data, \
     list(data.keys().difference(["Embarked","Cabin", "Survived", "PassengerId"])), \
                     "Embarked", poly_deg=5)

# replace null features in data set by predictions
data.loc[data.Embarked.isnull(), "Embarked"] = \
         embarked.Predict(data[data.Embarked.isnull()])
         
data.loc[data.Fare.isnull(), "Fare"] = \
         fare.Predict(data[data.Fare.isnull()])        
   
data.loc[data.Age.isnull(), "Age"] = \
         age.Predict(data[data.Age.isnull()])

cabin = Predictor(LogisticRegression(C=0.1), data, \
     list(data.keys().difference(["Cabin", "Survived", "PassengerId"])), \
                     "Cabin", ["Embarked"], poly_deg=5)
         
# replace null features in train set by predictions
train.loc[train.Embarked.isnull(), "Embarked"] = \
         embarked.Predict(train[train.Embarked.isnull()])
         
train.loc[train.Fare.isnull(), "Fare"] = \
         fare.Predict(train[train.Fare.isnull()])        
   
train.loc[train.Age.isnull(), "Age"] = \
         age.Predict(train[train.Age.isnull()])
         
train.loc[train.Cabin.isnull(), "Cabin"] = \
         cabin.Predict(train[train.Cabin.isnull()])
   

survived = Predictor(MLPClassifier(hidden_layer_sizes=(100,50,10,), max_iter=500), train, \
     list(train.keys().difference(["Survived", "PassengerId"])), \
                     "Survived", ["Embarked", "Cabin"])

# replace null features in test set by predictions
test.loc[test.Embarked.isnull(), "Embarked"] = \
         embarked.Predict(test[test.Embarked.isnull()])
         
test.loc[test.Fare.isnull(), "Fare"] = \
         fare.Predict(test[test.Fare.isnull()])        
   
test.loc[test.Age.isnull(), "Age"] = \
         age.Predict(test[test.Age.isnull()])
         
test.loc[test.Cabin.isnull(), "Cabin"] = \
         cabin.Predict(test[test.Cabin.isnull()])
   

#predict test set
test["Survived"] = survived.Predict(test)

test[["PassengerId","Survived"]].to_csv("solution.csv", index=False)