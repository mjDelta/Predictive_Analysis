# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 22:58:29 2017

@author: ZMJ
"""
import pandas as pd
import matplotlib.pyplot as plt
from math import exp

target_url="http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone=pd.read_csv(target_url,header=None,prefix="V")
#给DataFrame的列命名
abalone.columns=['Sex','Length','Diameter','Height',\
                'Whole Weight','Shucked Weight','Vidcera Weight',\
                'Shell Weight','Rings']

#获得描述性统计信息
summary=abalone.describe()

minRings=summary.iloc[3,7]
maxRings=summary.iloc[7,7]
nrows=len(abalone.index)

#标签归一化，以颜色深浅作为区别信息
for i in range(nrows):
  dataRow=abalone.iloc[i,1:8]
  labelColor=(abalone.iloc[i,8]-minRings)/(maxRings-minRings)
  dataRow.plot(color=plt.cm.RdYlBu(labelColor),alpha=0.5)
plt.xlabel("Attribute Index")
plt.ylabel("Attribute Value")
plt.show()

#标准正态分布的归一化后，利用sigmoid函数处理
meanRings=summary.iloc[1,7]
stdRings=summary.iloc[2,7]

for i in range(nrows):
  dataRow=abalone.iloc[i,1:8]
  labelColor=(abalone.iloc[i,8]-meanRings)/stdRings
  exp_color=1./(1.+exp(-labelColor))
  dataRow.plot(color=plt.cm.RdYlBu(exp_color),alpha=0.5)
plt.xlabel("Attribute Index")
plt.ylabel("Attribute Value")
plt.show()