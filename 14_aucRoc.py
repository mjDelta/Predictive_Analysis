# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 23:15:19 2017

@author: ZMJ
"""
import urllib2
import numpy 
import random
from sklearn import datasets,linear_model
from sklearn.metrics import roc_curve,auc
import pylab as pl

#自定义混淆矩阵:返回[tp,fn,fp,tn]
def confusionMatrix(predicted,actual,threshold):
  if len(predicted)!=len(actual):
    return -1
  tp=0.
  fp=0.
  tn=0.
  fn=0.
  for i in range(len(actual)):
    if actual[i]>0.5:
      if predicted[i]>threshold:
        tp+=1.0
      else:
        fn+=1.0
    else:
      if predicted[i]>threshold:
        fp+=1.0
      else:
        tn+=1.0
  rtn=[tp,fn,fp,tn]
  return rtn
  
target_url="https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"

data=urllib2.urlopen(target_url)

xList=[]
label=[]
for line in data:
  row=line.strip().split(",")
  if row[-1]=="M":
    label.append(1.0)
  else:
    label.append(0.)
  row.pop()
  floatRow=[float(num) for num in row]
  xList.append(floatRow)

indices=range(len(xList))
xListTest=[xList[i] for i in indices if i%3==0]
xListTrain=[xList[i] for i in indices if i%3!=0]
labelTest=[label[i] for i in indices if i%3==0]
labelTrain=[label[i] for i in indices if i%3!=0]
    
#将list转成矩阵，为后续的sklearn模型做数据格式准备
xTrain=numpy.array(xListTrain)
yTrain=numpy.array(labelTrain)
xTest=numpy.array(xListTest)
yTest=numpy.array(labelTest)



#训练线性回归模型
model=linear_model.LinearRegression()
model.fit(xTrain,yTrain)

#样本内数据预测
trainPredictions=model.predict(xTrain)
#生成混淆矩阵
confusionMatrixTrain=confusionMatrix(trainPredictions,yTrain,0.5)

#样本外数据预测
testPredictions=model.predict(xTest)
#生成混淆矩阵
confusionMatrixTest=confusionMatrix(testPredictions,yTest,0.5)

#绘制样本内的ROC曲线
fpr,tpr,thresholds=roc_curve(yTrain,trainPredictions)
roc_auc=auc(fpr,tpr)

pl.clf()
pl.plot(fpr,tpr,label="ROC curve area=%0.2f" %roc_auc)
pl.plot([0,1],[0,1],"k-")
pl.xlim([0.,1.0])
pl.ylim([0.,1.])
pl.xlabel("FPR")
pl.ylabel("TPR")
pl.title("In sample ROC")
pl.legend(loc="lower right")
pl.show()

fpr,tpr,thresholds=roc_curve(yTest,testPredictions)
roc_auc=auc(fpr,tpr)

pl.clf()
pl.plot(fpr,tpr,label="ROC curve area=%0.2f" %roc_auc)
pl.plot([0,1],[0,1],"k-")
pl.xlim([0.,1.0])
pl.ylim([0.,1.])
pl.xlabel("FPR")
pl.ylabel("TPR")
pl.title("Out sample ROC")
pl.legend(loc="lower right")
pl.show()