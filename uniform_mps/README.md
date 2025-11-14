# UniformMPS (TensorKit + OptimKit)

最小可跑通的 uMPS 基态项目骨架，按 A :: TensorMap(P ← W_L ⊗ W_R)（P=3，W_L=W_R=20）组织，后续可无缝迁移到 2D（CTMRG 环境 + 环转移）。

## 结构

- `src/UniformMPS.jl`：主模块入口
- `src/autodiff/utils.jl`：自动微分辅助工具（Wirtinger 导数等）
- `src/toolbox.jl`：局域两站点哈密顿量等物理量的收缩实现（不走 MPO）
- `src/optim/optim_ground_state.jl`：OptimKit 接口（LBFGS 起步）
- `examples/s1_heisenberg.jl`：spin-1 最近邻 Heisenberg 示例脚本
- `examples/tfim_spinhalf.jl`：spin-1/2 横场 Ising (TFIM) 示例脚本

## 依赖

建议 Julia ≥ 1.9：

```julia
using Pkg
Pkg.activate("uniform_mps")
Pkg.add(["TensorKit","TensorOperations","KrylovKit","OptimKit","Zygote"])
```

## 运行示例

```bash
julia uniform_mps/examples/s1_heisenberg.jl

# 或者运行 TFIM 示例
julia uniform_mps/examples/tfim_spinhalf.jl
```

> 需要切换不同自旋维度或 bond 维度时，可调用
> `UniformMPS.set_space_config!(; phys_dim=…, bond_dim=…)`，
> 或传入自定义的 `TensorKit.ComplexSpace` 对象。

> 说明：能量密度通过局域两站点哈密顿量直接评估，**不再构造 MPO**。`fixedpoints` 尚未注册 rrule；若需要端到端可微，请按 notes.md 2.6 的主丛/FS 拉回思路为固定点编写 rrule（隐式微分），或在优化阶段先用有限差分/近似自然梯度替代。

## 迁移到 2D 的钩子

- 把 `fixedpoints` 换成 `ctmrg_env(A)`（左/右环境）
- 把两站点收缩替换为 CTMRG 环境下的两点连通
- 把 `(I−E)^{-1}` 等线性算子换成环转移 `(I−T_ring)^{-1}` 的求解
- 保持 OptimKit 接口与 FS 度量/投影不变
