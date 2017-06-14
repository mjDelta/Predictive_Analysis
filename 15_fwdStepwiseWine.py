# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 23:03:47 2017

@author: ZMJ
"""
import numpy as np
from sklearn import linear_model
from math import sqrt
import matplotlib.pyplot as plt
import urllib2

def xattrSelect(x,idxSet):
  xOut=[]
  for row in x:
    xOut.append([row[i] for i in idxSet])
  return xOut

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
 
#一次增加一个属性
attributeList=[]
index=range(len(xList[1]))
indexSet=set(index)
indexSeq=[]
oosError=[]
for i in index:
  attSet=set(attributeList)  
  attTrySet=indexSet-attSet
  #将set转成list
  attTry=[li for li in attTrySet]
  errorList=[]
  attTemp=[]
  for iTry in attTry:
    attTemp=[]+attributeList
    attTemp.append(iTry)
    xTrainTemp=xattrSelect(xListTrain,attTemp)
    xTestTemp=xattrSelect(xListTest,attTemp)
    
    xTrain=np.array(xTrainTemp)
    yTrain=np.array(labelTrain)
    xTest=np.array(xTestTemp)
    yTest=np.array(labelTest)
    #使用sklearn中的linear_model
    model=linear_model.LinearRegression()
    model.fit(xTrain,yTrain)
    #计算rmsError
    rmsError=np.linalg.norm((yTest-model.predict(xTest)),2)/sqrt(len(yTest))
    errorList.append(rmsError)
    attTemp=[]
  iBest=np.argmin(errorList)
  attributeList.append(attTry[iBest])
  oosError.append(errorList[iBest])
print "Best attribute index:",
print attributeList
nameList=[names[i] for i in attributeList]
print "Best attribute name",
print nameList

x=range(len(oosError))
plt.plot(x,oosError,"k")
plt.xlabel("Number of Attribute")
plt.ylabel("Error (RMS")
plt.title("Forward Step Wise Regression")
plt.show()

#获取逐步向前回归的最佳属性
indexBest=oosError.index(min(oosError))
attributeBest=attributeList[1:(indexBest+1)]

xTrainList=xattrSelect(xListTrain,attributeBest)
xTestList=xattrSelect(xListTest,attributeBest)

xTrain=np.array(xTrainList)
xTest=np.array(xTestList)

#画柱状图
model=linear_model.LinearRegression()
model.fit(xTrain,yTrain)
errorVector=yTest-model.predict(xTest)
plt.hist(errorVector)
plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.show()

