# 方法、数据与实验依据

- [电子近视性](https://arxiv.org/abs/cond-mat/0503124)：局域近似有适用范围，不能据此假定所有体系都具有固定短截断。
- [CONQUEST 线性标度](https://conquest.readthedocs.io/en/latest/groundstate.html)：局域支持函数、密度核与误差控制。本库不是 CONQUEST 的复现。
- [SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)：有界有限记忆优化，最终投影驻点残差仍独立判断。
- [PySCF DFT](https://pyscf.org/user/dft.html)、[振动分析](https://pyscf.org/pyscf_api_docs/pyscf.hessian.html)：参考能量、力、密度与后续 Hessian/频率比较。
- [Paesani 水自电离数据](https://github.com/paesanilab/Data_Repository/tree/main/DC-r2SCAN_water_autoionization)：原始标签、作者划分和来源版本另行记录。
- [H2O-13](https://qmml.org/datasets.html)：Bartók 等，Phys. Rev. B 88, 054104 (2013)，DOI 10.1103/PhysRevB.88.054104。
- [DeePMD 数据格式](https://docs.deepmodeling.com/projects/deepmd/en/stable/data/data-conv.html)：数组与单位约定。
- [水自电离](https://pubmed.ncbi.nlm.nih.gov/11251111/)：稀有溶剂重排和氢键网络，不能由两水短轨迹推断体相 pKw。
- [Zundel 二聚体](https://arxiv.org/abs/1312.2897)：随 O–O 距离变化的质子势能面与几何。
- [质子化水簇的温度和核量子效应](https://doi.org/10.1038/s41467-023-42366-4)：质子共享、势垒和 Zundel/Eigen 分布。经典轨迹不能自动实现量子离域。
- [MACE](https://arxiv.org/abs/2206.07697)：多体等变模型对照背景；本库的矩收缩描述符不是 MACE。

这些来源支持研究路线和对照选择，不认证当前模型的整体化学精度。
