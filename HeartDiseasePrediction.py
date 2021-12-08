# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:30:46 2021

@author: cihat
"""
#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#2.veriOnisleme
#2.1 veri yükleme
veriler= pd.read_csv("heart.csv")
#veriler= pd.read_csv("veriler.csv")
age=veriler.iloc[:,0:1]
sex=veriler.iloc[:,1:2]

chest=veriler.iloc[:,2:3]
restingBP=veriler.iloc[:,3:6]
resting=veriler.iloc[:,6:7]
maxhr=veriler.iloc[:,7:8]
exercise=veriler.iloc[:,8:9]
oldpeak=veriler.iloc[:,9:10]
st_lope=veriler.iloc[:,10:11]
y=veriler.iloc[:,11:12]
Y=y.values
#Encoder:Kategorik -> Numeric

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#1,2,3 :LabelEncoder() 
ohe=OneHotEncoder()
le=LabelEncoder()

sex = pd.DataFrame(ohe.fit_transform(sex).toarray())
chest=pd.DataFrame(le.fit_transform(chest))
resting = pd.DataFrame(le.fit_transform(resting))
exercise = pd.DataFrame(ohe.fit_transform(exercise).toarray())
st_lope = pd.DataFrame(le.fit_transform(st_lope))

sex=sex.iloc[:,0]
exercise=exercise.iloc[:,0]
#DATAFRAME BİRLEŞTİRME
x=pd.concat([age,sex,chest,restingBP,resting,maxhr,exercise,oldpeak,st_lope],axis=1)
X=x.values


#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)


#verilerin ölçeklenmesi aynı dünyaya koyuyoruz verileri
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train= sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)


