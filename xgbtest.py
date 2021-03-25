# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 08:36:54 2020

@author: koachzz
"""
# First XGBoost model for Pima Indians dataset
import io_module
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def output_test_y(model):
    X_test = io_module.get_test_data()
    
    print("predicting...")
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    ans = []
    for i in range(len(predictions)):
        ans.append(['Y','N'][np.bool(predictions[i]==0)])
        
    with open('ans.csv','w',encoding="utf-8") as csvfile:
        for i in ans:
            csvfile.write(i+'\n')
    # print(predictions)
    print("predict done!")        

xtrain,ytrain = io_module.get_train_data()
X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.33, random_state=42)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

output_test_y(model)