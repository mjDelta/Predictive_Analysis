# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 09:17:58 2017

@author: ZMJ
"""
import pandas as pd
import matplotlib.pyplot as plot
from pandas import DataFrame
#pandas直接读取网络数据
target_url=("https://archive.ics.uci.edu/ml/machine-learning-"\
  "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
"""
pd.read_csv()：header=None表示不要将数据第一行作为列标签；prefix表示列标签的前缀为"V"
"""
rocks_mines=pd.read_csv(target_url,header=None,prefix="V")

#计算两两属性之间的相关性，corr直接计算两两属性之间的相关性
corMat=DataFrame(rocks_mines.corr())

#用热图展示属性的相关性，pcolor画热图
plot.pcolor(corMat)
plot.show()