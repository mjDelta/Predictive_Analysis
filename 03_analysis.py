# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 23:10:06 2017

@author: ZMJ
"""
import pandas as pd
import matplotlib.pyplot as plot
#pandas直接读取网络数据
target_url=("https://archive.ics.uci.edu/ml/machine-learning-"\
  "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
"""
pd.read_csv()：header=None表示不要将数据第一行作为列标签；prefix表示列标签的前缀为"V"
"""
rocks_mines=pd.read_csv(target_url,header=None,prefix="V")

dataColumn2=rocks_mines.iloc[:,1]
dataColumn3=rocks_mines.iloc[:,2]

plot.scatter(dataColumn2,dataColumn3)
plot.xlabel("Attribute 2")
plot.ylabel("Attribute 3")
plot.show()