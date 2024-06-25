Select Cell
==============

Select Cell的目的是计算出各个细胞的更详细信息，包括细胞位于的Spot、各种计算中得到的相关系数等。

Init Solution
----------------

.. code::python

  def init_solution(cell_type_num, # 比照ST_decon，但是数值由占比变成了该type细胞的数
    spot_idx, # spot列表，虽然貌似就是cell_type_num.index
    csr_st_exp, # ST_exp （稀疏）
    csr_sc_exp, # SC_exp （稀疏）
    meta_df,    # SC_meta
    trans_id_idx, # cell_<xxx>转<xxx>的表，虽然完全可以通过int(cell_idx[5:])进行转换
    T_HALF): # 传入的一个参数
    # 遍历所有Spot；将Spot的表达量作为向量去点乘单细胞基因表达矩阵，得到所有细胞关于该Spot的相关性
    # TODO: 这里好像把cell_type_num=0的细胞也乘进去了？不过性能损失其实一般
    # 排序；将相关性最大的cell选入该Spot
    # 返回初始解以及相关数据
    pass

Swap Selection
----------------

（对应reselect_cell，之前已经优化过了）

进行多次迭代，每次迭代仅从相邻Spot中互换一个细胞，并进行评分，根据分值决定是否采纳新方案，
如此迭代多轮后得到各个细胞较好的Spot分布情况。

评分标准：

1. Spot的表达量与选入Spot中细胞们表达量之和的相关系数
2. 相邻Spot之间interface，与相邻Spot中细胞们之间interface的相关系数

每次迭代中：

1. 从Spot中拿走一个细胞，然后邻近Spot中所有该cellType的其他细胞成为候选
2. 这些候选细胞的表达量向量合并成矩阵
3. 将它加上Spot中剩余细胞的表达量之和得到新矩阵，依次计算其与Spot表达量的相关系数
4. 选出相关系数最大的，该相关系数就是评分，若相关系数比换掉之前更小则不换