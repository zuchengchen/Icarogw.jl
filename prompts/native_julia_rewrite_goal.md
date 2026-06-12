# Codex Goal Prompt: Rewrite `../icarogw` As Native Julia `Icarogw.jl`

你正在 `/home/czc/projects/working/Icarogw.jl` 仓库中工作。目标是把相邻目录 `../icarogw` 中的 Python 版 `icarogw` 改写为一个高性能、原生 Julia 包。

这是一个长期迁移任务。请按阶段推进，每个阶段保持仓库可测试、可运行，并在稳定节点自动提交 Git commit。不要自动 push。

## 总目标

将 `../icarogw/icarogw` 中第一版范围内的功能重写为 native Julia 代码，形成标准 Julia 包 `Icarogw.jl`。

核心要求：

- 正式实现必须是原生 Julia。
- 禁止在正式包代码中使用 `PyCall.jl`、`PythonCall.jl`、`CondaPkg.jl` 或其他 Python bridge。
- 可以在开发/测试阶段临时运行 `../icarogw` Python 源项目生成参考数值 fixture，但这些调用不得进入正式包代码。
- 不要求 Julia 版调用与 Python 版完全相同的第三方包；优先选用成熟、可维护、支持 Julia 1.10 LTS 的 Julia 原生依赖。
- 性能优先，但以可维护、可验证的 Julia 风格实现为前提。
- 不做逐行翻译。允许按 Julia 的 multiple dispatch、`struct`、类型稳定数组布局、函数式接口、workspace/cache 等方式重新设计。
- 必须提供 Python-to-Julia API/功能映射文档，说明哪些功能已实现、重构、合并、排除或留待后续。

## 目标仓库与参考仓库

- 目标仓库：当前目录 `/home/czc/projects/working/Icarogw.jl`
- Python 参考项目：`../icarogw`
- 本地 Julia Dynesty 参考/集成包：`../Dynesty.jl`

优先读取本地源码：

1. 当前目标仓库
2. `../icarogw`
3. `../Dynesty.jl`

允许联网查询 Julia 原生依赖、官方文档、包文档、GitHub README、JuliaHub/General registry 信息。不要复制外部项目的大段代码；只参考 API、生态选择和文档。

## 第一版必须覆盖的功能范围

第一版必须全量覆盖 Python 版中属于以下功能域的内容。API 可以 Julia-native 重构，但功能不能只做少量 toy subset。

### 1. 宇宙学与转换

覆盖 `cosmology.py`、`conversions.py` 中与第一版相关的核心功能，包括但不限于：

- redshift 与 luminosity distance 转换
- `z -> dL`
- `dL -> z`
- comoving distance / comoving volume / `dVc/dz`
- detector-frame 与 source-frame 质量转换
- 质量、chirp mass、mass ratio、ISCO frequency 等转换
- magnitude/luminosity conversion
- modified-gravity cosmology wrappers 中不依赖后续 catalog/stochastic 的部分
- spin prior conversion 中属于 population inference/simulation 所需的部分

### 2. Population priors、rate models、wrappers

覆盖 `priors.py`、`wrappers.py`、`rates.py` 中第一版范围内的模型，包括但不限于：

- power-law mass models
- power-law + Gaussian peak
- broken power law
- multi-peak models
- mass-ratio models
- redshift evolution models
- CBC rate density evaluation
- spin distributions
- 条件分布、paired mass distribution、piecewise/bin models 中不依赖后续 catalog/stochastic 的部分

### 3. 层级贝叶斯 likelihood / population inference

重写 Python `likelihood.py` 中第一版范围内的 inference 能力，但不要模仿 `bilby.Likelihood` 继承结构。

必须支持：

- posterior-sample reweighting population likelihood
- injection / selection-effect correction
- Poisson rate likelihood
- shape-only likelihood
- no-event / upper-limit likelihood
- `../Dynesty.jl` 的原生接口示例
- diagnostics：
  - per-event effective sample size
  - injection effective sample size
  - `xi`
  - `N_expected`
  - 权重极端值、NaN、Inf、underflow 风险检查

后续才做：

- stochastic / galaxy catalog / EM counterpart 混合似然

### 4. Posterior samples 与 injections 数据容器

覆盖 `posterior_samples.py`、`injections.py` 中第一版范围内的功能：

- 组织 posterior samples
- 组织 injection sets
- 检查 prior/key/column 是否匹配
- 支持高性能 likelihood 所需的数组布局
- 支持从文件和内存对象构造数据容器
- 提供清晰的 validation 与 diagnostics

### 5. Simulation / mock data generation

覆盖 `simulation.py` 中第一版范围内的功能：

- 生成 mock source masses/redshifts
- SNR cut
- measurement noise
- quick posterior sample generation
- injection set generation
- seeded RNG 可复现
- 支持 smoke-test 规模与较大规模 benchmark

### 6. Examples / scripts / plotting / notebooks

必须提供：

- `examples/` 下可运行的 Julia examples
- 至少一个端到端 population inference example
- 至少一个 `../Dynesty.jl` integration example
- 基础 plotting，使用 `Plots.jl`
- `notebooks/` 或 notebook-oriented workflow 说明
- `examples/README.md`

不需要：

- Condor / HTCondor / 集群提交脚本生成
- Python 版 `utils.py` 中的 Condor helper 功能

## 后续功能：只做占位，不做核心实现

以下功能不属于第一版核心实现：

- galaxy catalog / dark siren / bright siren
- stochastic background / `Omega_GW`
- stochastic/catalog/EM counterpart 混合似然

可以创建克制的占位模块，例如 `Catalog.jl`、`Stochastic.jl`、`OmegaGW.jl`，但必须：

- 在 docstring 和文档中清楚标注 planned / not implemented in first version
- 相关 public function 若存在，必须抛出清楚错误
- 测试只验证错误信息清楚
- 不允许写未经验证的半成品伪实现

## 明确排除项

- 不实现 Condor / HTCondor 功能。
- 不支持 pickle。
- 第一版不强制支持 NPZ；`NPZ.jl` 可作为后续或迁移工具。
- 第一版不实现 CUDA/GPU 后端。
- 不移植 Python 的 `cupy_pal.py` 数组后端切换机制。
- 不自动 push 到远端。

## Julia 版本与依赖策略

目标 Julia 版本：

- 以 Julia 1.10 LTS 为最低/主要兼容目标。
- `Project.toml` 的 `julia` compat 应以 `1.10` 为基准。
- 不主动使用只在 Julia 1.11+ 才稳定可用的语法或标准库能力。

允许使用成熟 Julia 原生依赖。优先考虑但不限于：

- `SpecialFunctions.jl`
- `Distributions.jl`
- `StatsBase.jl`
- `QuadGK.jl` 或其他成熟积分包
- `Interpolations.jl`
- `Roots.jl`
- `HDF5.jl`
- `CSV.jl`
- `DataFrames.jl`
- `Tables.jl`
- `JLD2.jl`，可选，用于 Julia-native cache/checkpoint
- `FITSIO.jl` / HEALPix 相关 Julia 包，仅当第一版必要功能需要最小天图支持时使用
- `Plots.jl`
- `BenchmarkTools.jl`
- 可维护的性能辅助包，例如 `StaticArrays.jl`、`StructArrays.jl`、`PreallocationTools.jl`、`LoopVectorization.jl` 等；只在核心热路径有明确收益时使用

`../Dynesty.jl`：

- 不作为 `Icarogw.jl` 核心包的硬加载依赖。
- 核心包必须可在不加载 Dynesty 的情况下使用。
- 在 examples / tests / dev workflow 中可以通过本地路径 `../Dynesty.jl` 作为开发依赖使用。
- 提供采样器无关接口，使用户也可接其他 Julia 采样器。

## 包结构建议

创建标准 Julia 包结构。建议但可按实际实现调整：

```text
Project.toml
README.md
LICENSE
src/Icarogw.jl
src/Cosmology.jl
src/Conversions.jl
src/Priors.jl
src/Rates.jl
src/Likelihood.jl
src/DataContainers.jl
src/Simulation.jl
src/Plotting.jl
src/DynestyInterface.jl
src/Catalog.jl
src/Stochastic.jl
src/OmegaGW.jl
test/runtests.jl
test/reference/
examples/
examples/README.md
notebooks/
docs/
docs/api_mapping.md
docs/migration_notes.md
benchmark/
```

命名：

- 顶层 Julia module 名称：`Icarogw`
- 仓库名：`Icarogw.jl`
- 类型使用 Julia 风格 `CamelCase`，例如 `PowerLaw`, `PowerLawPeak`, `FlatLambdaCDM`, `PosteriorSamples`, `InjectionSet`
- 函数使用 Julia-native 风格，例如 `luminosity_distance`, `dvc_dz`, `source_to_detector`, `detector_to_source`, `loglikelihood`
- 对主要公共 API 可提供少量 Python-name alias，例如 `detector2source`，方便迁移
- 文档和新 examples 以 Julia-native API 为主
- 必须在 `docs/api_mapping.md` 中记录 Python 名称到 Julia 名称的映射

## 参数接口与采样接口

采用双层参数接口：

- 采样/性能主路径：`Vector{Float64}` 或 `AbstractVector{<:Real}` + parameter schema
- 人类友好入口：`NamedTuple`

必须设计 parameter schema，记录：

- 参数名
- 参数范围
- prior 类型
- prior transform
- 固定参数
- 默认值
- 单位或物理约定，若适用

建议接口：

```julia
schema = parameter_schema(model)
theta = prior_transform(schema, u)
named = unpack(schema, theta)
theta2 = pack(schema, named)
loglikelihood(model, data, theta)
loglikelihood(model, data, named)
```

要求：

- `loglikelihood(model, data, theta::AbstractVector)` 是热路径。
- `NamedTuple` 接口是便利层，不能成为性能瓶颈。
- 不要在热路径中反复构造 `Dict`、`DataFrame` 或动态类型对象。
- batch likelihood 输入可为矩阵，必须在 docstring 中固定维度约定。
- diagnostics 返回结构化 Julia 类型，不靠打印。

## Likelihood 性能与并行策略

默认策略：

- 单次 `loglikelihood` evaluation 默认单核、确定性、低分配。
- 默认 likelihood 热路径中不要调用 `Threads.@threads`。
- 不依赖隐藏的全局可变状态。
- 避免隐藏 BLAS 多线程依赖；如涉及 BLAS，请文档说明。
- 固定输入和参数时，`loglikelihood` 必须 deterministic。
- 优先优化单核性能：
  - 类型稳定
  - 预分配
  - cache-friendly 数组布局
  - `@views`
  - log-space arithmetic
  - 减少 allocation
  - 合理使用 interpolation/cache/workspace

并行策略：

- 并行放在采样器或外层 orchestration 层，而不是单次 likelihood 内部。
- 提供接口让 `Dynesty.jl` 或用户代码并发评估多个参数点。
- 数据容器和模型应尽量 immutable 或支持 read-only 并发调用。
- 提供 opt-in batch API，例如：

```julia
loglikelihood_batch(model, data, theta_matrix; parallel=false)
loglikelihood_batch(model, data, theta_matrix; parallel=true)
```

- `parallel=true` 是显式选择，不得改变默认 scalar likelihood 的单核行为。
- simulation、benchmark、数据预处理等离线任务可以使用多线程，但必须显式 opt-in。

## IO 与数据格式

第一版必须支持：

- Julia in-memory containers：
  - `NamedTuple`
  - `Dict`
  - `Tables.jl` compatible table
  - `DataFrame`
- `HDF5.jl`：
  - posterior samples
  - injections
  - 大数组
- `CSV.jl` + `DataFrames.jl`：
  - 轻量示例
  - 用户可读表格

可选：

- `JLD2.jl` 用于 Julia-native cache/checkpoint

后续/迁移工具：

- `NPZ.jl`

不支持：

- pickle

FITS/HEALPix：

- 主要留给后续 catalog 功能。
- 如果第一版必要功能碰到基础 skymap/sky pixel 支持，可以做最小 native 支持，但不要扩大到完整 catalog 迁移。

## Dynesty.jl 集成

核心包不应要求 `using Dynesty` 才能加载。

必须提供：

- 采样器无关的 `loglikelihood` 和 `prior_transform` 接口。
- 一个 `Dynesty.jl` integration layer 或 example。
- 使用本地 `../Dynesty.jl` 的开发/示例说明。

示例至少展示：

- 定义 population model
- 定义 parameter schema / prior transform
- 生成或读取 toy posterior samples
- 生成或读取 toy injections
- 调用 `Dynesty.jl` 进行 nested sampling
- 输出 posterior samples / evidence / diagnostics
- 使用 `Plots.jl` 画基础结果图

如果 `../Dynesty.jl` 不可用或 API 与预期不同，不要猜。读取其源码或文档；必要时让 example 清楚跳过，并在文档中说明。

## 测试与数值验证

由于 `../icarogw` 没有完整现成测试，必须建立迁移验证体系。

### 1. 单元测试

覆盖：

- 标量输入
- 数组输入
- shape / broadcasting
- 边界值
- invalid input
- PDF normalization
- rate / likelihood finite checks
- parameter schema pack/unpack/prior transform
- diagnostics

### 2. Python 参考数值回归

开发阶段允许临时调用 `../icarogw` Python 源项目生成小型参考 fixture。

建议生成并保存到 `test/reference/`：

- cosmology distances
- mass conversions
- prior pdf/logpdf
- rate evaluations
- posterior/injection likelihood toy case
- simulation sanity reference

正式 Julia tests 只能读取 fixture，不得在测试时依赖 Python runtime。

### 3. 统计/蒙特卡洛测试

对随机 simulation 和 sampling helper：

- seeded RNG 可复现
- 分布均值/方差范围
- histogram 或 KS-like sanity checks
- selection cut 后计数范围
- 不要求逐点相等

### 4. 性能基准

提供 `benchmark/` 或 `perf/`：

- likelihood evaluation
- posterior weights
- injection weights
- cosmology interpolation
- simulation mock generation

使用 `BenchmarkTools.jl` 记录 allocations 和时间。测试不一定强制性能阈值，但必须能让后续比较。

### 数值容差建议

- 普通代数/转换函数：`rtol` 约 `1e-8` 到 `1e-10`
- 涉及积分/插值/特殊函数：根据 Python/Julia 实现差异设合理容差，通常 `1e-5` 到 `1e-7`
- likelihood toy case：尽量严格
- 较复杂 fixture：使用合理相对容差并记录原因

## 文档要求

第一版必须提供实用型文档，不强制搭建 `Documenter.jl` 网站。

必须包括：

### `README.md`

- 项目定位
- 安装方式
- quickstart
- 最小 population inference 示例
- 与 Python 版 `icarogw` 的关系
- license / attribution
- 第一版覆盖范围
- 暂不支持范围

### `docs/api_mapping.md`

- Python 模块/函数/类 到 Julia API 的映射
- 状态：
  - implemented
  - renamed
  - merged
  - excluded
  - planned
- 明确标注：
  - Condor excluded
  - catalog planned
  - stochastic / OmegaGW planned

### `docs/migration_notes.md`

说明：

- native Julia 设计差异
- parameter schema
- 数据容器
- likelihood 接口
- single-core likelihood / sampler-level parallelism
- batch likelihood
- Dynesty 集成
- IO 格式支持
- 性能策略
- 已知差异

### Docstrings

所有 public 类型和 public 函数必须有 docstring，说明：

- 参数
- 返回值
- 单位
- 数值约定
- 公式或参考行为，若适用

### Examples 文档

`examples/README.md` 必须说明每个 example：

- 做什么
- 怎么运行
- 预期输出
- 需要哪些可选依赖

## License 与 attribution

Python 源项目 `../icarogw` 声明 license 为 `EUPL-1.2`。

要求：

- 尊重并保留原项目 attribution。
- 默认沿用或兼容原项目 `EUPL-1.2`，除非仓库所有者之后明确选择其他合法方案。
- 添加 `LICENSE`。
- 在 `README.md` 和 `docs/migration_notes.md` 中说明 Julia 版与 Python 版 `icarogw` 的关系。
- 不要擅自改成 MIT、Apache 或其他 license。

## 执行顺序

不要一开始横向铺满所有模块。按下面顺序推进：

### 阶段 1：Inventory 与 mapping

- 扫描 `../icarogw`
- 列出第一版必须迁移、后续、排除的函数/类
- 创建 `docs/api_mapping.md` 初稿
- 识别 Python 中依赖 catalog/stochastic/Condor 的部分并标记

### 阶段 2：Julia 包骨架

- 创建 `Project.toml`
- 创建 `src/Icarogw.jl` 和核心模块
- 创建 `test/runtests.jl`
- 创建 `examples/`、`docs/`、`benchmark/`
- 添加 README/LICENSE 初稿

### 阶段 3：最小端到端 vertical slice

先打通一个可运行 workflow：

- 简单 cosmology
- 一个 mass prior
- 一个 redshift/rate model
- toy posterior samples
- toy injections
- shape-only likelihood
- Poisson likelihood
- no-event likelihood
- diagnostics
- `Dynesty.jl` example
- `Plots.jl` 基础 plot

这个 slice 必须能跑通 tests 和 example。

### 阶段 4：横向补全第一版功能域

补全：

- cosmology
- conversions
- priors
- wrappers/models
- rates
- posterior samples
- injections
- simulation
- likelihood
- plotting
- examples/notebook workflow

按 Python-to-Julia mapping 持续更新状态。

### 阶段 5：测试、benchmark、文档收口

- 生成/整理 Python reference fixtures
- 完善单元测试
- 完善统计测试
- 完善 benchmarks
- 完善 docs
- 跑 `Pkg.test()`
- 跑关键 examples
- final commit

每个阶段都要保持仓库可测试、可运行。不要等所有文件写完才第一次跑测试。

## 完成定义

任务完成时必须满足：

- `Pkg.test()` 通过。
- 至少一个端到端 example 可运行，例如：

```bash
julia --project examples/basic_population_inference.jl
```

- 至少一个 `Dynesty.jl` integration example 可运行；如果本地 `../Dynesty.jl` 不可用或 API 不匹配，example 必须给出清楚跳过说明。
- 核心 public API 有 docstrings。
- `docs/api_mapping.md` 完整列出 first-version-scope Python API 的迁移状态。
- `docs/migration_notes.md` 说明设计差异和暂不支持功能。
- `benchmark/` 或 `perf/` 至少包含 likelihood、weighting、simulation 的 benchmark。
- 正式包代码不依赖 Python bridge。
- `README.md` 包含安装、quickstart、example、license/attribution。
- 后续功能占位模块清楚标注 planned/not implemented。
- Condor 功能明确排除。
- 工作区最后应是清楚状态，并有 final commit。

## Git 提交要求

必须自动 commit，但不要自动 push。

建议多阶段 milestone commits：

1. `scaffold Julia package and migration docs`
2. `implement vertical-slice population inference`
3. `port core cosmology conversions priors and rates`
4. `implement data containers simulations examples and plotting`
5. `add tests benchmarks docs and final polish`

要求：

- 每次 commit 前运行对应阶段 tests 或 smoke tests。
- 不提交明显坏掉的中间状态。
- commit message 清楚描述阶段成果。
- 不自动 `git push`。
- 如果工作区一开始有用户未提交改动，先识别并避免混入不相关变更；必要时说明。

## 遇到歧义时的处理规则

- 遇到科学公式歧义或 Python 版行为不清楚时，不要凭空猜；优先读 Python 源码、生成参考 fixture，并在文档记录差异。
- 遇到 Julia 生态包缺口时，优先写小型 native fallback，并在文档记录原因。
- 遇到 `../Dynesty.jl` API 不明确时，读取本地源码并适配真实 API。
- 不为了表面兼容 Python API 写低性能动态代码。
- 不实现未要求的一整块 catalog/stochastic/Condor 功能。
- 不自动 push。

## 最终回复要求

完成后向用户汇报：

- 已实现的主要功能
- 哪些 Python 模块/功能已迁移
- 哪些功能明确排除或留待后续
- tests/examples/benchmarks 运行结果
- commit 列表
- 任何已知数值差异或风险
