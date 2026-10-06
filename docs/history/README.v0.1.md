# uni-chem CA · v0.1

**跨格点原子元胞 × 同步并行演化 × 紧凑电子态 × 全局物理场。**

这是一个可以运行、训练、测试和扩展的研究元模型。它不是 DFT 求解器，
也不是已经训练好的水溶液反应模型。附带 H/O 参数和能量泛函仅用于检验软件与数值结构。

## 一个原子不是一个像素

这里有两种不同的“单元”：

- **原子级扩展元胞**：原子核/有效核芯、连续位置、动量、原子轨道、电子云尺度和原子身份。
- **空间格点**：对世界进行离散积分、沉积电荷和计算场的有限体积单元。

一个原子覆盖很多空间格点；一个空间格点可以同时接收多个原子的电子云贡献。
邻接原子的电子子空间允许相干混合，不是把独立刚性小球放到网格上。

```text
连续核位置 + 原子轨道模板 + 环境相关电子云尺度
                    │
          多对多原子—格点支持映射
                    │
      跨原子的电子轨道系数 + 自旋占据
                    │
          有约束的 1-RDM / 价电子密度
                    │
             全局 Poisson 场
                    │
      同一总能量 → 自动微分 → 所有核的力
                    │
        同步电子更新 / 同步核更新 / latent CA
                    │
             统一提交新的 World
```

网格变细意味着同一电子云覆盖更多格点，**不应该使原子物理半径变小**。
云的大小由基函数宽度和电子态决定；球形支持域只是有限计算范围。
s/p 电子云可以各向异性，原子间允许重叠。小于网格可分辨尺度的配置会报错，不会退化成一像素原子。

## 安装与运行

使用 Python 3.10+，建议 CPU 验证时使用 float64。当前交付包实际测试环境见
`docs/VALIDATION.md`；声明的最低依赖版本不等于逐一验证过所有版本。

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -e ".[dev]"
python -m pytest -q
unichem demo --steps 3 --output outputs/demo
unichem train-demo --epochs 25 --output outputs/training
python examples/check_resolution.py
```

依赖已经存在但不能联网时，可直接在仓库根目录运行：

```bash
PYTHONPATH=src python -m unichem demo --steps 3
PYTHONPATH=src python -m unichem train-demo --epochs 25
python -m pytest -q
```

`PYTHONPATH=src` 的写法适用于 POSIX shell；Windows 可先安装 editable 包。
`demo` 输出 `world_initial.npz`、`world_final.npz`、`fields_final.npz` 和 JSON 诊断。
`train-demo` 使用**合成标签**，输出真实训练后的权重和训练/留出集损失，不能作为化学精度证明。

## 最小 API

```python
import torch
from unichem import Grid, make_world, Evaluator, SynchronousEngine
from unichem.runtime.electrons import ParallelElectronicRelaxer

# 所有长度单位为 bohr；同一个原子通常覆盖数百至数千个格点。
world = make_world(
    ["O", "H", "H"],
    [[8.03, 8.07, 8.11], [9.63, 8.07, 8.11], [7.63, 9.57, 8.11]],
    grid=Grid(shape=(28, 28, 28), spacing=(0.6, 0.6, 0.6)),
    dtype=torch.float64,
)

evaluator = Evaluator()  # 未标定的演示能量，不能用于真实反应结论。
engine = SynchronousEngine(
    evaluator,
    dt=0.1,  # 原子时间单位，不是 fs。
    relaxer=ParallelElectronicRelaxer(max_iterations=3),
)

result = engine.step(world)
next_world = result.world  # world 保持不变，输出张量不与输入共享存储。
print(result.diagnostics["voxels_per_atom"])
print(result.diagnostics["electron_integral"])
print(result.diagnostics["electrons_converged"])
```

返回 `electrons_converged=False` 时，不能把本步的力称为已收敛 Born–Oppenheimer 力。
真实研究应开启 `require_electronic_convergence=True` 并建立对应参考势的收敛标准。
电子弛豫是数值求解过程，不是电子的真实时间动力学。

## 量子结构保留到什么程度

原子轨道保存壳层标签 `n`、角动量 `ell`、实球谐分量 `s/px/py/pz` 和云宽度。
`px/py` 不是确定磁量子数的复本征态。示范 Gaussian 也不是完整氢样径向函数。
每个空间电子轨道保存两种自旋的占据数，单自旋占据在 `[0,1]` 内。

用 AO 样本矩阵 A 和格点体积 ΔV 定义实际使用的离散重叠矩阵：

```text
S = Aᵀ A ΔV
Cᵀ S C = I
P = C diag(n_up + n_down) Cᵀ
rho[g] = Σ_p (n_up[p] + n_down[p]) · (A[g,:] C[:,p])²
Tr(P S) = Σ_g rho[g] ΔV = N_valence
```

`C` 的一列可以同时出现在多个原子的 AO 行上。因此电子对可跨原子分布，
保留成键/反键干涉所需的结构，而非仅有一个“原子电荷”标量。
`bond=True`、`molecule_id`、反应方程表不是引擎输入。

默认氧使用 +6 有效核芯和六个价电子，因此 O/H/H 示例是 **8 个价电子**，
不是全电子水分子的十电子计算。全部电荷沉积与这个约定一致。

## 并行约定

每个同步阶段只读取一个完整旧快照，用批量张量生成候选状态，
经过全局场/约束屏障后统一提交。不允许前一个原子的更新影响同阶段后一个原子。
支持域、电子密度沉积、核更新和 latent stencil 都遵守这个约定。

固定的三空间方向循环和全局自洽迭代不等于按原子顺序更新。
参考电子求解器有全局 Cholesky 正交化和全局线搜索，静电有全局 FFT。
所以这不是严格“只有最近邻通信”的经典 CA，而是**全局场耦合的同步扩展元胞系统**。
同步语义不等于已经实现多 GPU 分布式计算。

## 神经网络与 latent

`SharedScalarResidual` 学习共享的局部标量/邻域能量残差；力统一来自同一总能量的导数。
`UnrolledPredictor` 可展开同步电子弛豫，使电子密度损失也能反传到能量网络。
`ConservativeLatentCA` 实现共享面通量和局部记忆更新，保持各 latent 通道的全局和。
`SharedElectronicPreconditioner` 让场和 latent 调节电子求解的数值预条件。

**v0.1 不让未经校准的 latent 网络额外施加核力或宣称学会真实慢反应。**
改变预条件器只能改变求解路径；在同一收敛精度下，不应改变同一能量的解。
未来的动力学粗粒化、热浴、记忆核或稀有事件加速需要单独验证。

## 目录

```text
src/unichem/
  state.py, grid.py, species.py, factory.py
  context.py, evaluate.py, checkpoint.py, training.py, cli.py
  quantum/  support.py, basis.py, density.py, observables.py
  fields/   poisson.py
  models/   energy.py, latent.py
  runtime/  electrons.py, engine.py
examples/   演化、训练、分辨率检验
tests/     几何、量子约束、场、同步性、导数、训练、检查点
docs/      架构、物理边界、扩展协议、训练数据规范、实际验证记录
```

## 扩展性边界

v0.1 的原子—格点支持关系已经是稀疏多对多结构，但量子计算采用
**有明确容量上限的稠密参考实现**。默认最多 128 个 AO、800 万个 AO 网格样本元素。
邻居对构造也是带上限的 O(N²) 参考实现；没有宣称可以运行百万原子。

该实现用于建立可替换的能量/求解组件和数值正确性基线。真正的大规模版本需要
块稀疏占据子空间、分块积分、邻居链表、域分解与分布式场求解。
详见 `docs/EXTENDING.md`，包括当前接口哪些部分必须一起升级。

物理依据及不能作出的结论见 `docs/PHYSICS_SCOPE.md`。本项目使用 MIT 许可证。
