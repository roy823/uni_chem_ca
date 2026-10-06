# uni-chem CA · v0.3

跨格点原子、块稀疏局域电子表示、同能量梯度力与同步演化的研究内核。

**当前真实化学验证未通过。** 独立水对相对势能曲线误差约 0.77 eV，平均力误差约 2.57 eV/Å。三个短轨迹未出现质子归属变化，不能据此测量自由能、速率或体相 pKw。

进一步实验方案见 [EXPERIMENTS.zh-CN.md](docs/EXPERIMENTS.zh-CN.md)。它区分数值复现、分子性质复现与化学涌现，并先检验原生同步引擎和不同力定义。

新一轮已完成 162 点真实 DFT、从零量子训练和保留面验收：Zundel 测试面曲线误差约 0.51–0.72 eV，三轴极化率低估约 98%–99.7%，探索性水轨迹约 5.8 fs 即偏离水几何域。粗网格整体平移能量误差可达 0.51 eV，加密后相位误差降低但 DFT 力精度未恢复。应先校准电子能量与离散化，再扩展反应统计。

## 运行

```powershell
python -m pip install -e ".[dev,data]"
python -m pytest -q
unichem ca-controls --steps 20
unichem reference-screen
unichem design-reference
unichem two-water
```

依赖已存在时可以先设置 `$env:PYTHONPATH = 'src'`，再用 `python -m unichem` 代替安装后的命令。默认水对实验先检查独立精度；显式 `--exploratory` 才运行未校准轨迹，并明确记录其性质。

## 电子表示

`ElectronState.kernel` 保存原子图上的局域核坐标，物理自旋密度核由局域正矩阵函数产生。重叠、动能和密度核使用 `BlockMatrix`，不构造系统级稠密 C/P/S，不调用全局 Cholesky/Cayley。两个标量化学势约束全体系电子数。

Pauli 约束来自整体矩阵函数构造，不能由各 patch 分别合法推断。多项式 halo、填充和度量误差明确报告；预算不足直接失败。当前未验证十万原子液体、多 GPU 或严格最近邻通信。

## 真实数据

公开数据与生成数据按方法分别保存，避免混合绝对能量：

| 来源 | 内容 |
|---|---|
| 作者公开 DC-r2SCAN 数据 | 热水、自电离相关和场相关构型；重复样本与未知场约定分别隔离 |
| H2O-13 | 6,900 个 PBE0/aug-cc-pV5Z 水单体构型，含拉伸和高能区 |
| 本次 PySCF 参考 | 41 个 PBE0/aug-cc-pVDZ 构型，含水、H₃O⁺、OH⁻、Zundel、Eigen、水对及外场 |
| 新一轮冻结设计参考 | 162 个真实 PBE0/aug-cc-pVDZ 标签：水局部势能面、三轴外场、117 点对称 Zundel 扫描；82/35/45 训练/验证/测试 |

原始下载哈希和来源记录在 `data/*manifest.json`；生成参考包含能量、力、AO 1-RDM 与匹配网格的价电子密度。参考基组和模型基组不同，不直接比较 AO 系数。

```powershell
python examples/fetch_references.py
unichem prepare-reference --generated
unichem train-reference --dataset data/ho_pbe0 --output outputs/ho_final_quantum
```

生成真实标签需要 Linux/macOS PySCF 2.10：`python examples/generate_ho_dft.py --check-field`。本次在已有 Ubuntu/WSL 中实际运行。几何设计文件自身没有标签；新一轮 162 个任务已实际完成 SCF、能量/力和密度采集。

来源：[Paesani 数据仓库](https://github.com/paesanilab/Data_Repository/tree/main/DC-r2SCAN_water_autoionization)、[H2O-13](https://qmml.org/datasets.html)、[PySCF](https://pyscf.org/user/dft.html)。未从缺失的许可声明推断再分发授权。

## 演化与实验

`LocalQuantumResidual` 学习共享多体几何描述符、局域核种子、云宽度与标量残差。H/O 研究基组包含径向 s 通道与完整 p 通道；参数和能量闭合仍需要验证。

`KernelUnrolledPredictor` 的力包含整个有限电子映射的响应。`SynchronousEngine` 持续推进 World，并使用当前独立电子坐标下的力。两者的物理定义需要独立核查，不能自动认为相同。

`runtime/potential.py` 只有核能量/力回调与 BAOAB 更新，没有反应规则。随机数以持续原子 ID 和全局步号寻址；重启必须保留这些状态。观察者在 `analysis/`，只读已生成轨迹。

详细数学、范围与结果：[DESIGN.zh-CN.md](docs/DESIGN.zh-CN.md)、[TRAINING.md](docs/TRAINING.md)、[PHYSICS_SCOPE.md](docs/PHYSICS_SCOPE.md)、[VALIDATION.md](docs/VALIDATION.md)。旧稠密/流式/Cayley 路径和旧 checkpoint 迁移已删除。

新一轮纯几何对照已复现水单体：键长误差 0.0023 Å、键角误差 0.082°、三个谐振频率误差均小于 1%，三个步长的各 1 ps NVE 轨迹稳定且能量误差按二阶收敛。原生 `SynchronousEngine` 的 2 fs 同能量对照与回调积分器坐标差约 5.3e-15 bohr，重启逐位一致。这是几何能量与核更新的限定域验证，电子态在此对照中不参与能量，不能据此声称量子化学或自电离已经涌现。
