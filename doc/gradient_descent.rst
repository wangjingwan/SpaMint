Gradient Descent
==================

之前的Swap Selection仍然离最优解有距离，因此进一步使用梯度下降来计算。

梯度下降的基本思路就是定义损失函数表示拟合程度，然后让损失函数越小越好，
直到达到极小值。在这个项目中，每次迭代中一共有五个拟合函数与五个损失函数，
其目标是修改结果中各个细胞的表达量，每次迭代中将修改的表达量应用到新的sc_exp中。

拟合函数1
------------

拟合函数2
------------

拟合函数3
------------

由于每次迭代都会修改表达量的值，而细胞坐标是根据表达量算出Affinity再据其估算的。
为了计算term3，每次迭代开头都会进行Affinity Embedding计算坐标，再根据坐标计算
细胞KNN。可以简化成从相邻Spot中找N个Affinity最大的细胞，省去每次的Embedding。

拟合函数4
------------

拟合函数5
------------