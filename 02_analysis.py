# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 22:36:24 2017

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

"""
dataframe中定位到元素用：iat
"""
for i in range(200):
  #根据标签指定两种不同的颜色

  if rocks_mines.iat[i,60]=="M":
    c="red"
  else:
    c="blue"
  dataRow=rocks_mines.iloc[i,0:60]
  dataRow.plot(color=c)

plot.xlabel("Attribute Index")
plot.ylabel("Attribute Values")
plot.title("Parallel Coordinates")
plot.show()
