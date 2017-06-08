# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 22:09:10 2017

@author: ZMJ
"""
import pandas as pd
import matplotlib.pyplot as plt

target_url="http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone=pd.read_csv(target_url,header=None,prefix="V")
#给DataFrame的列命名
abalone.columns=['Sex','Length','Diameter','Height',\
                'Whole Weight','Shucked Weight','Vidcera Weight',\
                'Shell Weight','Rings']

#获得描述性统计信息
summary=abalone.describe()

#画出前八个属性的盒状图
array=abalone.iloc[:,1:9].values
plt.boxplot(array)
plt.xlabel("Attribute Value")
plt.ylabel("Quartile Ranges")
plt.show()

#画出前七个属性的盒状图，因为第八个的取值范围过大，严重压缩了前七个的盒状图，此为解决方法一：直接去除
array2=abalone.iloc[:,1:8].values
plt.boxplot(array2)
plt.xlabel("Attribute Value")
plt.ylabel("Quartile Ranges")
plt.show()

#对数据进行归一化，此为解决方法二
normal_data=abalone.iloc[:,1:9]

for i in range(8):
  #直接利用上边的描述性统计信息中的均值方差
  mean=summary.iloc[1,i]
  std=summary.iloc[2,i]
  normal_data.iloc[:,i:i+1]=(normal_data.iloc[:,i:i+1]-mean)/std

array3=normal_data.values
plt.boxplot(array3)
plt.xlabel("Attribute Value")
plt.ylabel("Quartile Ranges")
plt.show()