Prepare for Start
==================

此为程序执行之前的一些准备工作。主要围绕构造函数展开，其中prep函数被我转为私有了
（本来就只会在构造的时候调用的）

程序接受的输入有：

- Spot的数据

  * ST_coord: spot的坐标，demo中是19x10的方格，但是少了spot_113
  * ST_exp: spot的表达量，189x[基因数]
  * ST_decon: spot中各个cellType占比，189x[细胞种类数]

- Single Cell的数据

  * SC_exp: 单个细胞的表达量，[细胞数]x[基因数]
  * SC_meta: 单细胞的基本信息：集群与cellType
  * SC_smurf: SC_exp + rand(shape) * 10，作用不明

- 其他参数

  * LR_df: 和计算Affinity有关的LR，df大概表示这是DataFrame吧

预处理
----------

（其实我认为这些应该都由那个对象的构造函数来完成）

预处理的代码主要在preprocess.py - prep_all_adata()中

.. code:: python

  def prep_all_adata(sc_exp = None, st_exp = None, sc_distribution = None, 
      sc_meta = None, st_coord = None, lr_df = None, SP = 'Human'):
    # 1. data_clean(): 删除从未表达的基因
    # 2. denoise_gene(): 只留下各个数据集都涉及的基因（交集），弃掉其他基因
    # LR_df中只留下使用到的LR对
    # scale数据到SUM SUM似乎写着1e4而不是1
    # make_adata略
    pass

.. note::

  LR_df筛查在之后的步骤中重复计算了。

构造SpaMint对象
-----------------

构造函数就是将参数赋给类的各个成员，除了要注意weight对应ST_decon表。
然后将之前的AnnData又转换成DataFrame，寻找一次SpotKNN，删除没有近邻的spot

HVG和SVG不知道是啥，先不管，
最后计算ST的Affinity profile (L*R+R*L)
