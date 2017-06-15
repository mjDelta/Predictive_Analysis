# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 23:39:22 2017

@author: ZMJ
"""
import urllib2
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

target_url="https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"

data=urllib2.urlopen(target_url)

xList=[]
labels=[]
for line in data:
  row=line.strip().split(",")
  if(row[-1]=="M"):
    labels.append(1.0)
  else:
    labels.append(0.)
  row.pop()
  floatRow=[float(num) for num in row]
  xList.append(floatRow)

indices=range(len(xList))
xListTest=[xList[i] for i in indices if i%3==0]
xListTrain=[xList[i] for i in indices if i%3!=0]
labelsTest=[labels[i] for i in indices if i%3==0]
labelsTrain=[labels[i] for i in indices if i%3!=0]

xTrain=np.array(xListTrain)
xTest=np.array(xListTest)
yTrain=np.array(labelsTrain)
yTest=np.array(labelsTest)

alphaList=[0.1**i for i in [-4,-3,-2,-1,0,1,2,3,4,5]]
aucList=[]
for a in alphaList:
  model=linear_model.Ridge(alpha=a)
  model.fit(xTrain,yTrain)
  fpr,tpr,threshlods=roc_curve(yTest,model.predict(xTest))
  roc_auc=auc(fpr,tpr)
  aucList.append(roc_auc)

x=[-4,-3,-2,-1,0,1,2,3,4,5]
plt.plot(x,aucList)
plt.xlabel("-log(alpha)")
plt.ylabel("AUC")
plt.show()

indexBest=aucList.index(max(aucList))
alpha=alphaList[indexBest]
print alpha
model=linear_model.Ridge(alpha=alpha)
model.fit(xTrain,yTrain)

plt.scatter(model.predict(xTest),yTest,s=100,alpha=0.3)
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()
