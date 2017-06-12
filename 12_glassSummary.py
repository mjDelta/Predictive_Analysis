# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:22:03 2017

@author: ZMJ
"""
import pandas as pd
import matplotlib.pyplot as plot

target_url="http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

glass=pd.read_csv(target_url,header=None,prefix="v")
glass.columns=["Id","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type"]

summary=glass.describe()

glassNormalized=glass.iloc[:,1:]
ncol=len(glassNormalized.columns)
summary2=glassNormalized.describe()

for i in range(ncol):
  mean=summary2.iloc[1,i]
  std=summary2.iloc[2,i]
  glassNormalized.iloc[:,i:i+1]=(glassNormalized.iloc[:,i:i+1]-mean)/std

array=glassNormalized.values
plot.boxplot(array)
plot.xlabel("Attribute Index")
plot.ylabel("QuratileRanges- Normalized")
plot.show()
