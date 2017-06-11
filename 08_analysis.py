# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:00:23 2017

@author: ZMJ
"""
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

target_url="http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone=pd.read_csv(target_url,header=None,prefix="V")
abalone.columns=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

#计算相关矩阵
corMat=DataFrame(abalone.iloc[:,1:9].corr())

#利用热图可视化相关矩阵
#与分类的热图展示相关矩阵不同的一点：可以把标签的相关性绘制出来，因为回归的y值连续
plot.pcolor(corMat)
plot.show()