# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:30:09 2017

@author: ZMJ
"""
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
from math import exp

target_url="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

wine=pd.read_csv(target_url,header=0,seq=";")
summary=wine.describe()

nrow=len(wine.index)
ncol=len(wine.columns)

meanTaste=summary.iloc[1,ncol-1]
stdTaste=summary.iloc[2,ncol-1]

for i in range(nrow):
  dataRow=wine.iloc[i,:ncol-1]
  target=(wine.iloc[i,ncol-1]-meanTaste)/stdTaste
  labelColor=1.0/(1.0+exp(target))
  dataRow.plot(color=plot.cm.RdYlBu(labelColor),alpha=0.5)
plot.xlabel("Attribute Index")
plot.ylabel("Attribute Value")
plot.show()

#将每一列属性正则化再画平行坐标图
wineNormalized=wine

for i in range(ncol):
  mean=summary.iloc[1,i]
  std=summary.iloc[2,i]
  wineNormalized.iloc[:,i:i+1]=(wineNormalized.iloc[:,i:i+1]-mean/std)
for i in range(nrow):
  dataRow=wineNormalized.iloc[i,:ncol-1]
  target=(wineNormalized.iloc[i,ncol-1]-meanTaste)/stdTaste
  labelColor=1.0/(1.0+exp(target))
  dataRow.plot(color=plot.cm.RdYlBu(labelColor),alpha=0.5)
plot.xlabel("Attribute Index")
plot.ylabel("Attribute Value")
plot.show()

#用正则化的数据画关系热图
corMat=DataFrame(wineNormalized.corr())  
plot.pcolor(corMat)
plot.show()
  