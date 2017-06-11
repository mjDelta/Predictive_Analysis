# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:13:09 2017

@author: ZMJ
"""
import pandas as pd
import matplotlib.pyplot as plot

target_url="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

wine=pd.read_csv(target_url,header=0,seq=";")

summary=wine.describe()

wineNormalized=wine
ncols=len(wineNormalized.columns)

for i in range(ncols):
  mean=summary.iloc[1,i]
  std=summary.iloc[2,i]
  wineNormalized.iloc[:,i:i+1]=(wineNormalized.iloc[:,i:i+1]-mean)/std
  
array=wineNormalized.values
plot.boxplot(array)
plot.xlabel("Attribute Index")
plot.ylabel("Quartile Ranges - Normalized")
plot.show()
