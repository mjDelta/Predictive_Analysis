# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 22:16:28 2017

@author: ZMJ
"""
import pylab
import scipy.stats as stats
import urllib2

#使用urllib获得网上数据
target_url=("https://archive.ics.uci.edu/ml/machine-learning-"\
  "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
data=urllib2.urlopen(target_url)
#把数据转成list
xList=[]
labels=[]
for line in data:
  row=line.strip().split(",")
  xList.append(row)
  
#统计有多少条数据，每条数据有多少个属性
nrow=len(xList)
ncolumn=len(xList[0])

#针对第三列数据做分析：分位数图（QQ图）
col=3
colData=[]
for row in xList:
  colData.append(float(row[col]))
"""
stats.probplot:此处分位图（又叫Q-Q图）展示了数据集的百分位边界与高斯分布的同样的百分位边界的对比；
解释：若服从高斯分布，则画出来的应该是一条直线；如图，明显尾部数据
"""
stats.probplot(colData,dist="norm",plot=pylab)
pylab.show()
