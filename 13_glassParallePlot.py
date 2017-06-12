# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:22:02 2017

@author: ZMJ
"""
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

target_url="http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

glass=pd.read_csv(target_url,header=None,prefix="v")
glass.columns=["Id","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type"]

glassNormalized=glass

ncol=len(glass.columns)
summary=glassNormalized.describe()

for i in range(ncol-1):
  mean=summary.iloc[1,i]
  std=summary.iloc[2,i]
  glassNormalized[:,i:i+1]=(glassNormalized.iloc[:,i:i+1]-mean)/std

nrow=glass.index

for i in range(nrow):
  rowData=glassNormalized.iloc[i,:ncol-1]
  target=glassNormalized.iloc[i,ncol-1]/7.0
  rowData.plot(color=plot.cm.RdYlBu(target),alpha=0.5)
  
plot.xlabel("Attribute Index")
plot.ylabel("Attribute Value")
plot.show()

corMat=DataFrame(glassNormalized.iloc[:,1:ncol-1].corr())
plot.pcolor(corMat)
plot.show()