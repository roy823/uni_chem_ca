# uni-chem CA · v0.5

## 局域 CA 本体与整体场分层

主方向是固定状态大小、固定邻域和固定更新轮数的元胞执行。整体模型在本体外预测电子环境、能量和核力；本体读取同一个旧快照，执行局域电子输运与核更新，不运行全体系电子 SCF。

`LocalCellEngine` 已实现固定大小的局部自旋约化密度、两元胞通道与对称凸混合。电子数守恒与局部 Pauli 占据边界由局域规则保持，不依赖全局电子数校正。这是显式均场闭合，尚未保留完整跨元胞量子关联；不等同于已经验证的通用局域量子化学模型。

```bash
python examples/benchmark_local_ca.py --device cuda --waters 100 1000 10000 --steps 20
python examples/benchmark_local_ca.py --device cpu --waters 10 100 1000
```

已实际运行到 10,000 个水坐标构型、30,000 个原子元胞；RTX 4060 Laptop GPU 上 1,000 / 10,000 个构型的本体耗时约 0.0505 / 0.4955 秒每步，约 16.82 / 16.52 μs 每元胞每步。固定的是单元计算代价，同一块 GPU 的总耗时仍随工作量增加。

本轮输入场是未训练的数值测试场，核力为零；验证执行容量、电子数、占据边界与局域传播，不验证真实水反应。结果在 `outputs/local_ca_capacity/`。整体场模型和化学精度的验收是下一层任务。

## 独立 HF 物理参照

保留 v0.4 的原子量子 patches、零温密度核净化、自洽 Fock 求解与同步核演化，作为小体系物理参照。

v0.4 参照已从底层替换 v0.3 的软占据、±8 辅助坐标、演示能量和单步网络电子路径。所有原子使用同一组物理规则，输入只有元素、位置、基组、总电荷、自旋及外场；没有键表、反应规则或学习能量残差。

当前物理闭合是全电子 Hartree–Fock。PySCF/libcint 只提供原子块积分，本项目自行计算密度核、自洽电子态和同步演化；生产路径没有系统级稠密 C/P/S、Cholesky 或量子算子的全局对角化。

## 实际进展

| 无网络验证 | 结果 |
|---|---|
| 11 个同算子单电子/HF 参照 | 全部通过；最大能量差约 2.8e-14 Hartree |
| 水的核力与同基组独立 HF 参考 | 最大分量误差约 2.5e-10 原子单位 |
| 更完整 cc-pVDZ 水基组 | 能量差约 4.3e-13 Hartree、核力差约 1.7e-11 原子单位 |
| 无网络水几何与振动 | 自行优化得到稳定结构，频率与同方法参考相对差约 1.6e-7 |
| 原生水 NVE | 0.2/0.1/0.05 fs 步长各 10 fs；能量误差约 1.33e-5/3.34e-6/8.35e-7 eV，二阶收敛 |
| 有隙局域算子 64–512 站点 | 密度块约 1,774–20,192，展示受控稀疏表示；不代表分子积分已达到线性成本 |

这验证了选定 HF/基组层面的底层求解与分子性质复现。完整基组、电子相关、核量子效应、正确反应自由能/速率和大型体相性能仍需独立验证。

## 使用

基础代数、缓存积分验证可在 Windows 运行；实时物理积分需要 Linux/macOS 的 PySCF，Windows 使用已有 WSL 环境。

```bash
python -m pip install -e ".[dev,physics]"
python -m pytest -q
unichem evaluate --xyz examples/water.xyz --basis sto-3g
unichem simulate --xyz examples/water.xyz --steps 20 --dt-fs 0.05
```

已有依赖时可设置 `PYTHONPATH=src`，使用 `python -m unichem`。公开参考生成与数据下载脚本继续保留；它们不是核心运行的训练步骤。

```bash
python examples/water_dft_modes.py
python examples/generate_quantum_oracles.py
unichem validate-core --reference data/quantum_oracles
python examples/validate_integrals.py
python examples/validate_physical_core.py --duration-fs 10
python examples/validate_sparse_scaling.py
```

## 核心结构

- `molecule.py` 保存原子身份、位置、动量、质量、基组以及两个物理自旋密度核。
- `quantum/purification.py` 用残差控制 Newton–Schulz 度量变换和零温 TC2 净化，没有固定软占据底噪。
- `runtime/scf.py` 统一检查自洽交换子、密度变化、电子数、幂等性和物理 Fock 占据一致性。
- `quantum/gaussian.py` 提供动能、真实核吸引、Hartree、精确交换和包含 Pulay 项的解析核力。
- `runtime/molecular.py` 从同一电子能量推进所有原子，电子未收敛时不提交核状态。

稀疏压缩报告实际丢弃块的行范数界，并继续检查最终物理残差；容量超限直接失败。当前精确四中心积分供应器有显式预算，尚未认证大型致密水或多 GPU 扩展性。

v0.4 检查点只保存物理密度核，不迁移旧辅助坐标或旧网络权重。历史报告在 `docs/history/v0.3/` 和原输出目录，原源码快照为 `outputs/v03_source_snapshot.zip`。

## 水簇实验与动态三维网页

`examples/experiment_water_clusters.py` 测量 1–4 个水分子的积分、自洽、核力与完整时间步，并运行中性、高能随机初态、离子控制与共享质子轨迹。只读电子键指标、片段电荷和持续时间用于观察；不反馈反应规则。

```bash
python examples/experiment_water_clusters.py benchmark --threads 1
python examples/experiment_water_clusters.py benchmark --threads 4 --waters 1 2 3 4
python examples/experiment_water_clusters.py trajectory --scenario shared_proton --duration-fs 20
python examples/export_water_viewer.py
npm install --prefix web
python -m http.server 8765 --bind 127.0.0.1 --directory web
```

浏览器打开 `http://127.0.0.1:8765/`，可旋转、缩放、选择氢原子、播放、暂停与逐帧查看真实轨迹。网页显示距离、电子键指标、能量误差和计算成本；播放速度不是量子计算速度。

本轮实际完整时间步约为 0.12 / 0.94 / 3.50 / 9.64 秒（1 / 2 / 3 / 4 个水分子，单线程、0.1 fs 步长）。四线程仅得到约 1.09–1.19 倍观测加速。5 个水分子触及默认积分预算；尚不支持数百或上千水分子的完整物理实验。

详细记录：[架构](docs/ARCHITECTURE.zh-CN.md)、[物理范围](docs/PHYSICS_SCOPE.md)、[验证](docs/VALIDATION.md)、[实验](docs/EXPERIMENTS.zh-CN.md)。
