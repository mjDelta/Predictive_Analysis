# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 23:31:11 2017

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

target=[]
for i in range(208):
  if rocks_mines.iat[i,60]=="M":
    target.append(1.0)
  else:
    target.append(0.0)
"""
分析第36个属性与标签之间的关系，因为前面的平行分位图中30-40之间有明显差距
"""
dataRow=rocks_mines.iloc[:208,35]
plot.scatter(dataRow,target)
plot.xlabel("36th Attribute Value")
plot.ylabel("Label Value")
plot.show()


