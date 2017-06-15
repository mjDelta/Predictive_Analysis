# Predictive_Analysis
《Python机器学习预测分析核心算法》的实现记录</br>

分析部分：</br>
1.分位数图（QQ图）检测异常点（与正态分布对比的QQ图）</br>
![image](https://github.com/mjDelta/Predictive_Analysis/blob/master/imgs/QQPlot.png)</br>
2.盒状图检测异常点（多分类问题异常点的检测不同于二分类问题）</br>
![image](https://github.com/mjDelta/Predictive_Analysis/blob/master/imgs/boxPlot.png)</br>
3.平行坐标图，热图显示属性与目标之间的相关性（数据归一化后做平行坐标图，热图的效果更好一点）</br>
![image](https://github.com/mjDelta/Predictive_Analysis/blob/master/imgs/parallePlot.png)
![image](https://github.com/mjDelta/Predictive_Analysis/blob/master/imgs/heatPlot.png)</br>

模型评估部分：</br>
1.ROC&AUC曲线:展示了样本内ROC曲线，样本外的ROC曲线（可以看出样本内的AUC值更大，但是实际预测中往往使用样本外的AUC衡量）</br>
![image](https://github.com/mjDelta/Predictive_Analysis/blob/master/imgs/auc_insample.png)
![image](https://github.com/mjDelta/Predictive_Analysis/blob/master/imgs/auc_outsample.png)</br>

过拟合问题的处理：</br>
1.使用逐步向前回归控制过拟合（示例：红酒打分回归问题）</br>
逐步回归中rms误差随属性增加的变化曲线</br>
![image](https://github.com/mjDelta/Predictive_Analysis/blob/master/imgs/fwdstep.png)</br>
红酒打分问题的效果图</br>
![image](https://github.com/mjDelta/Predictive_Analysis/blob/master/imgs/fwdstepscatter.png)
![image](https://github.com/mjDelta/Predictive_Analysis/blob/master/imgs/fwdstephist.png)</br>
2.使用岭回归控制过拟合（示例：红酒打分回归问题），正则化</br>
逐步回归中rms误差随属性增加的变化曲线</br>
![image](https://github.com/mjDelta/Predictive_Analysis/blob/master/imgs/ridge.png)</br>
