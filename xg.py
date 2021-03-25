# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:04:00 2020

@author: USER
"""

import io_module
import numpy as np
import xgboost

from sklearn.model_selection import train_test_split

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
    print(predictions)
    print("predict done!")        

xtrain,ytrain = io_module.get_train_data()
X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.33, random_state=42)

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.04
params['max_depth'] = 20
params['learning_rate'] = 0.0001

d_train = xgboost.DMatrix(X_train, label=y_train)
d_valid = xgboost.DMatrix(X_test, label=y_test)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model = xgboost.train(params, d_train, 100, watchlist, early_stopping_rounds=100, verbose_eval=10)

y_pred = model.predict(xgboost.DMatrix(X_test))

print("Accuracy: ", str(sum(y_test == (y_pred > 0.5))/y_test.shape[0]))

output_test_y(xgboost.DMatrix(model))