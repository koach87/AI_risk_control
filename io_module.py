# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:51:32 2020

@author: peter
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing

TRAIN = r'X_Y_train.csv' # train path
# XTEST = r'X_test.csv' #train input
# TRAIN = r'X_Y_train_.csv' # train path
XTEST = r'X_test.csv' #train input
outputColums = ['loan_status']
inputColumns = [ 
                'loan_amnt',#借款金額和G(funded_amnt)&(funded_amnt_inv)值都一樣不知道差別固先合併
                "int_rate",#貸款利率 ％ to float
                'installment',#貸款人每月須償還金額
                "grade",#lc貸款等級,暫時和m(sub_grade)合併[e,d,c,b,a]
                "emp_length",#工作時間1~10年區分大於10年=10 小於1年=1
                'home_ownership',#房屋種類[MORTGAGE,RENT,OWN,OTHER]底壓,出租,擁有,其他
                "annual_inc",#年收入
                "verification_status",#收入是否被驗證？verified,source verified=1 not verified = 0
                "pymnt_plan",#貸款計畫 y or n
                'delinq_2yrs',#兩年內預期30天以上的次數
                'fico_range_low',#信用分數 5/range
                'inq_last_6mths',#過去6個月貸款數量
                'mths_since_last_delinq',#上次拖欠債務以來的月數
                'open_acc',#未清貸款數
                'pub_rec',#公開貶損數
                'revol_util',#循還線利用率
                'total_acc',#信用額度總數
                'initial_list_status',#貸款初始狀態（w,f)
                'out_prncp_inv',#為償還的本金總額
                'total_pymnt_inv',#已付款總額
                
                # 壞掉俱樂部:
                "term",#還款期數 36 or 60期 ｛0:36,1:60｝
                "purpose",#貸款類別 credit_card,bebt_consolidation,major_purchse.......
                'dti',#債務/收入
                # 'emp_title'暫pass 職業
                    
                 ]
scalar = {'loan_amnt':"NON",
          "term":"Onehot",
          'int_rate':"NON",
          'installment':"NON",
          "grade":"Onehot",
          "emp_length":"Onehot",
          'home_ownership':"Onehot",
          "annual_inc":"NON",
          "verification_status":"Onehot",
          "pymnt_plan":"Onehot",
          "purpose":"Onehot",
          'dti':'NON',
          'delinq_2yrs':'NON',
          'fico_range_low':'NON',
          'inq_last_6mths':"NON",
          'mths_since_last_delinq':'NON',
          'open_acc':'NON',
          'pub_rec':'NON',
          'revol_util':'NON',
          'total_acc':'NON',
          'initial_list_status':"NON",
          'out_prncp_inv':'NON',
          'total_pymnt_inv':'NON'
          }   
    
# scalar = {'loan_amnt':"Average",
#           "term":"Onehot",
#           'int_rate':"NON",
#           'installment':"Average",
#           "grade":"Onehot",
#           "emp_length":"Onehot",
#           'home_ownership':"Onehot",
#           "annual_inc":"Average",
#           "verification_status":"Onehot",
#           "pymnt_plan":"Onehot",
#           "purpose":"Onehot",
#           'dti':'Average',
#           'delinq_2yrs':'Average',
#           'fico_range_low':'Average',
#           'inq_last_6mths':"Average",
#           'mths_since_last_delinq':'Average',
#           'open_acc':'Average',
#           'pub_rec':'Average',
#           'revol_util':'NON',
#           'total_acc':'Average',
#           'initial_list_status':"NON",
#           'out_prncp_inv':'Average',
#           'total_pymnt_inv':'Average'
#           }

def total_pymnt_inv(data):
    data = data.fillna(0)
    return data

def out_prncp_inv(data):
    data = data.fillna(0)
    return data

def annual_inc(data):
    data = data.fillna(0)
    data = data.to_numpy()
    data[data>500000] = 500000
    data = data.reshape(data.shape[0],1)
    return data

def initial_list_status(data):
    data = data.fillna('f')
    return (np.array(data)!='f').astype(int)
    
def total_acc(data):
    data = data.fillna(0)
    return data

def revol_util(data):
    data = int_rate(data)
    return data

def pub_rec(data):
    data = data.fillna(0)
    return data

def open_acc(data):
    data = data.fillna(0)
    return data

def mths_since_last_delinq(data):
    data = data.fillna(0)
    return data

def fico_range_low(data):
    data = data.fillna(0)
    return data

def term(data):#36期=0 60期＝1
    data = data.fillna('36')
    data = (np.array(data)!='36').astype(int)
    return np.array(data,dtype='int64')

def int_rate(data): #去％ / 100
    data = data.fillna('0')
    data = data.str.strip('%')
    data = np.array(data, dtype = 'float64')
    # print(data)
    return data

def grade(data): #grade E = 0 ,D = 1, C = 2,B = 3, A = 4
    data = data.fillna('G')
    data = data.str.replace('A','6')
    data = data.str.replace('B','5')
    data = data.str.replace('C','4')
    data = data.str.replace('D','3')
    data = data.str.replace('E','2')
    data = data.str.replace('F','1')
    data = data.str.replace('G','0')
    return np.array(data,dtype='int64')

def emp_length(data): #year<=1 :1 >=10 : 10 處理有點久
    data = data.fillna('1')
    data = data.str.strip('<+years ')
    return np.array(data,dtype='int64')        
  
def home_ownership(data): #own = 3 mortgage =2 rent =1 other =0
    data = data.fillna("OTHER")
    data = data.str.replace('OWN','3')
    data = data.str.replace('MORTGAGE','2')
    data = data.str.replace('RENT','1')
    data = data.str.replace('OTHER','0')
    data = data.str.replace('ANY','0')
    data = data.str.replace('NONE','0')
    return np.array(data,dtype='int64')    
 
def verification_status(data): #Not Verified = 0 , Source Verified & Verified =1
    data = data.fillna('Not Verified')
    data = data.str.replace('Not Verified','0')
    data = data.str.replace('Source Verified','1')
    data = data.str.replace('Verified','1')
    return np.array(data,dtype='int64')

def pymnt_plan(data):#y= 1 n =0
    data = (np.array(data)=='y').astype(int)
    return  np.array(data,dtype='int64')

def purpose(data):
    #xtrain["emp_length"] = xtrain["emp_length"].fillna('other')
    data = data.fillna('other')
    test = ['debt_consolidation', #0 
            'credit_card',        #1
            'other',              #2
            'home_improvement', 
            'major_purchase', 
            'car', 
            'small_business', 
            'vacation', 
            'medical', 
            'moving', 
            'house', 
            'renewable_energy',
            'wedding',
            'educational']  #13
    for i in range(len(data)):
        for j,item in enumerate(test):
            if item in data[i]:
                data[i] = j
                break
    return np.array(data,dtype='int64')

def Average(data):
    data = np.array(data,dtype='float64')
    data = data.reshape(data.shape[0],1)
    data = preprocessing.MinMaxScaler().fit_transform(data)
    return np.array(data,dtype='float64')

def Onehot(data):
    data = pd.get_dummies(data, drop_first=True)
    data = data.to_numpy()  
    return data

def NON(data):
    data = data.to_numpy()
    data = data.reshape(data.shape[0],1)
    return data

def normalized(data):
    temp = np.zeros(shape=(len(data),1))
    for i in scalar:
        if i in data:
            temp = np.append(temp,globals()[scalar[i]](data[i]),axis=1)
    return temp

def trans(data):
    translate = [ 
                    "term",
                    "int_rate",
                    "grade",
                    "emp_length",
                    'home_ownership',
                    "verification_status",
                    "pymnt_plan",
                    "purpose",
                    'fico_range_low',
                    'mths_since_last_delinq',
                    'open_acc',
                    'pub_rec',
                    'revol_util',
                    'total_acc',
                    'initial_list_status',
                    'out_prncp_inv',
                    'total_pymnt_inv',
                    'annual_inc',
                 ]
    for i in inputColumns:
        if i in translate:
            # print("translating: " + i)
            data[i] = globals()[i](data[i])
            
def testla(data,s):
    print(data[s])
    data[s]=globals()[s](data[s])
    print((data[s]))
    return np.array(data[s],dtype='int64')
    
def Y_N(data):
    return  (np.array(data)=='Y').astype(int)

def get_train_data():#return (xtrain,ytrain)
    
    print("reading xtrain...")
    xtrain = pd.read_csv(TRAIN, usecols = inputColumns, na_values=['?'])
    print("reading ytrain...")
    ytrain = pd.read_csv(TRAIN, usecols = outputColums)
    
    # xtrain = xtrain.dropna(subset=inputColumns)
    
    print("translating...")
    trans(xtrain)
    print("normalizing...")
    xtrain = normalized(xtrain)
    xtrain = preprocessing.MinMaxScaler().fit_transform(xtrain)
    print("spliting ytrain...")
    ytrain = Y_N(ytrain['loan_status'])
    print("get_train_data train data Done!")
    return(xtrain,ytrain)

def get_test_data():
    
    print("reading xtest...")
    xtest =  pd.read_csv(XTEST,usecols =inputColumns)
    print("translating...")
    trans(xtest)
    xtest = normalized(xtest)
    print("normalizing...")
    xtest = preprocessing.MinMaxScaler().fit_transform(xtest)
    return xtest

if __name__ == '__main__':
    x,y = get_train_data()
    xt = get_test_data()