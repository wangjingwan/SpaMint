Affinity Embedding
====================

这是关于affinity embedding步骤的思路整理过程，
以及对所接触的代码附上类型注解

该步骤发生于Gradient Descent开始运行之前。前一步Cell Selection建立了细胞与Spot的对应，
这一步的目标则是进一步计算sc_coord（单细胞坐标； ``np.ndarray`` 二维点集）

- 按注释，在一些测序方法中无法得知细胞的确切坐标，另外已经按固定坐标划定一个个Spot并选出了Spot中含有哪些细胞；
- 每个细胞对每个基因的表达量不同，因此将各个基因的表达量组在一起形成一个元组（向量），表达该细胞的各个表达量
- 每个Spot也能算出这样一个基因表达向量
- 根据每个细胞的相关性估计它的坐标

.. code:: python

  # 这是目前版本的代码
  def aff_embedding(alter_sc_exp: DataFrame, # cell x gene 矩阵
    st_coord: DataFrame, # 每个spot的x y坐标
    sc_meta: DataFrame, # 每个cell的对应Spot、cellType等数据
    lr_df: DataFrame, # 定义L R配对的表
    save_path: str, # 保存sc_coord到哪个文件？ 存疑
    left_range = 1,
    right_range = 2,
    steps = 1,
    dim = 2,
    verbose = False):
    pass

首先运行sc_adj_cal（意思应该是sc adjacent cal，算相邻细胞），先看目标，该函数\
返回去掉空spot后的st_coord（暂时没用上）、一个不知道什么sc_dis_mat、
一个不知道为什么分成12块保存的细胞x细胞所有距离的矩阵

其中：

- 执行sc_prep，输入st_coord和sc_meta，返回去掉空spot后的st_coord，
  以及初步的sc_coord（每个cell坐标均为spot坐标）
  (sc_prep中写了太多没必要的操作，已优化)
- 算出各个spot中相隔最小的x，可能相隔y也一样，这个就是unit_len
- 算出所有细胞间距离矩阵 (???) (细胞的坐标现在就是spot坐标还算这个大矩阵啊)
- 计算neigh(bor)，这是一个二维数组，里面是相距这个细胞距离小于unit*2的所有细胞 (...)
- 之前又有个细胞x细胞的矩阵，矩阵中0表示两个细胞不是相距，1表示两个细胞相距
- 总之，返回这样的表示相距的矩阵与距离矩阵（ans）

总之这是sc_adj_cal，算出了各个细胞自己几个相邻spot的细胞，用一个矩阵表示这种相邻关系；还算出了距离矩阵

再看下一步chunk_calc_aff，返回的是affinity矩阵

- 先从大的LR对应表中只挑出需要用到的
- 构造LxR和RxL所需的数组，但排除重复的
- 获得基因矩阵的转置 gene*cell, （A）获得所有细胞的L，（B）获得所有细胞的R
- 点乘score对角矩阵和A，转置得出新的A，以便接下来的矩阵乘法
- 点乘A和B获得aff矩阵，乘sc_dis_mat过滤掉不是相邻细胞的aff (...)
- 虽然还是有点看不懂，但显然乘了个上一步那个dis_mat，里面肯定很多很多0啊，说明莫名多计算了很多很多次
- 返回affinity矩阵

至此已经产生优化点1：

- 既然只要计算每个细胞与自己相邻细胞的aff，那就要避免矩阵乘法
- aff矩阵本身返回稀疏矩阵，感觉不用改，主要是中间的矩阵运算能删的就删
- 既然如此，上一步也没必要算那么大一个distance_matrix了，可以先算出一个spot相邻表（类似邻接表的结构），
  再据此快速得出sc_dis_mat
- 但由于后面还需要用到ANS矩阵，仍需充分讨论

继续看算出aff矩阵后接下来的，aff矩阵非0的减0.1，aff全部加0.1，aff用1填充对角线（这样的话恐怕已经不稀疏了），
进入embedding步骤。这一步接受aff矩阵和之前那个很大的距离矩阵，算出sc坐标

- 将过于接近0的aff值都改为0.1，将对角线都改成0，将非0值全部取倒数
- 利用umap对aff进行降维，然后基于之前的距离矩阵评估，毕竟要得到的是sc坐标，降维是肯定要降维的
- 同时在多层循环中不断修改umap的参数，生成各种不同的备选sc坐标
- 基于coord_eva(luate)函数不断对降维得到的东西评分，用评分最高的作为sc坐标
  - 基于这样的coord计算所有细胞间距离，将距离矩阵与之前那个ans距离矩阵计算相关系数，这就是评分

为了理解之前对aff矩阵的系列操作，需要先了解umap的基本原理。
为什么评分的计算量会如此大？