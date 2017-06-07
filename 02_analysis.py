# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 22:36:24 2017

@author: ZMJ
"""
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
#pandas直接读取网络数据
target_url=("https://archive.ics.uci.edu/ml/machine-learning-"\
  "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
"""
pd.read_csv()：header=None表示不要将数据第一行作为列标签；prefix表示列标签的前缀为"V"
"""
rocks_mines=pd.read_csv(target_url,header=None,prefix="V")

#查看头五条尾五条数据
print rocks_mines.head()
print rocks_mines.tail()

#查看描述统计信息
summary=rocks_mines.describe()
print summary

