# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 00:32:45 2017

@author: ZMJ
"""
import numpy as np
from sklearn import linear_model
from math import sqrt
import matplotlib.pyplot as plt
import urllib2

target_url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data=urllib2.urlopen(target_url)
xList=[]
labels=[]
names=[]
firstLine=True
for line in data:
  if firstLine:
    names=line.strip().split(";")
    firstLine=False
  else:
    row=line.strip().split(";")
    labels.append(float(row[-1]))
    row.pop()
    floatRow=[float(num) for num in row]
    xList.append(floatRow)
    #print len(row)

#划分训练集，测试集
indices=range(len(xList))
xListTest=[xList[i] for i in indices if i%3==0]
xListTrain=[xList[i] for i in indices if i%3!=0]
labelTest=[labels[i] for i in indices if i%3==0]
labelTrain=[labels[i] for i in indices if i%3!=0]

xTrain=np.array(xListTrain)
xTest=np.array(xListTest)
yTrain=np.array(labelTrain)
yTest=np.array(labelTest)

alphaList=[0.1**i for i in range(10)]
rmsError=[]
for a in alphaList:
  model=linear_model.Ridge(alpha=a)
  model.fit(xTrain,yTrain)
  rmsError.append(np.linalg.norm((yTest-model.predict(xTest)),2)/sqrt(len(xTest)))

x=range(len(rmsError))
plt.plot(x,rmsError)
plt.xlabel("-log(alpha)")
plt.ylabel("Error (RMS)")
plt.title("Ridge Regression")
plt.show()

indexBest=rmsError.index(min(rmsError))
alph=alphaList[indexBest]
model=linear_model.Ridge(alpha=alph)
model.fit(xTrain,yTrain)
errorVector=yTest-model.predict(xTest)
plt.hist(errorVector)
plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.title("Histogram")
plt.show()

plt.scatter(model.predict(xTest),yTest,s=100,alpha=0.1)
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()