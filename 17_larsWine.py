# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:14:14 2017

@author: ZMJ
"""
import urllib2
from math import sqrt
import matplotlib.pyplot as plt

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

#标准化数据以及标签
nrows=len(xList)
ncols=len(xList[0])

#计算均值，标准差
xMeans=[]
xStd=[]
for i in range(ncols):
  mean=sum(xList[:][i])/nrows
  xMeans.append(mean)
  std=sqrt(sum([(xList[j][i]-mean)**2 for j in range(nrows)])/nrows)
  xStd.append(std)

xNormalised=[]
for i in range(nrows):
  rowNormalised=[(xList[i][j]-xMeans[j])/xStd[j] for j in range(ncols)]
  xNormalised.append(rowNormalised)
  
meanLabel=sum(labels)/nrows
stdLabel=sqrt(sum([(labels[i]-meanLabel)**2 for i in range(nrows)])/nrows)
yNormalised=[(num-meanLabel)/stdLabel for num in labels]
            
beta=[0.0]*ncols

betaMat=[]
betaMat.append(list(beta))

nSteps=350
stepSize=0.004

for i in range(nSteps):
  residuals=[0.]*nrows
  for j in range(nrows):
    y=sum([xNormalised[j][k]*beta[k] for k in range(ncols)])
    residuals[j]=yNormalised[j]-y
             
  corr=[0.]*ncols
  for j in range(ncols):
    corr[j]=sum([xNormalised[k][j]*residuals[k] for k in range(nrows)])/nrows
  iStar=0
  corrStar=corr[0]
  
  for j in range(1,ncols):
    if abs(corrStar)<abs(corr[j]):
      iStar=j
      corrStar=corr[j]
  beta[iStar]+=stepSize*corrStar/abs(corrStar)
  betaMat.append(list(beta))
  
for i in range(ncols):
  coefCurve=[betaMat[k][i] for k in range(nSteps)]
  xaxis=range(nSteps)
  plt.plot(xaxis,coefCurve)

plt.xlabel("Step Taken")
plt.ylabel("Coeffient Values")
plt.title("LARS PLOT")
plt.show()

     

        
