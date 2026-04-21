# Part6-Core

本仓库对应论文中“外卖平台骑手行为与系统演化”实验的核心代码，包含内容：

- `simulation/` 与相关运行脚本：从原始 `SocialInvolution` 示例中整理出的、适合公开展示的轻量化模拟实现

当前仓库保留论文实验所需的核心流程，使代码结构更清晰、可多次运行、可直接分析。

## 仓库内容

### 1. 真实实验数据分析

- `analyze_experiments.py`：统计骑手与系统级指标
- `analyze_thoughts.py`：分析 thought 文本与绩效关系
- `1_analyze_thoughts_with_llm.py`：使用大模型抽取 thought 关键词
- `analyze_intention_behavior.py`：分析意图与行为变化的关系
- `4_generate_experiment_heatmaps.py`：生成轨迹热力图
- `generate_evolution_sankey.py`：生成意图演化桑基图
- `generate_core_figure.py`：生成论文核心组合图

### 2. 重构后的模拟模块

- `simulation/order.py`：订单对象定义
- `simulation/order_generator.py`：订单时序生成
- `simulation/rider.py`：骑手行为逻辑
- `simulation/platform.py`：平台状态与指标更新
- `simulation/city.py`：模拟环境调度器
- `simulation/individual_cal.py`：骑手个体指标
- `simulation/sys_cal.py`：平台系统指标
- `simulation/dispatch.py`：订单分配与简单路径逻辑

### 3. 模拟运行与结果分析

- `run_simulation.py`：执行单次或多次模拟
- `analyze_simulation.py`：分析多次模拟结果
- `analyze_multi_source.py`：将模拟结果与 `exp1` 到 `exp5` 做对比

## 目录说明

```text
part6-core/
├── simulation/               # 重构后的模拟代码
├── simulation_results/       # 多次模拟输出结果
├── analysis_results/         # 分析图表与汇总结果
├── run_simulation.py         # 模拟入口
├── analyze_simulation.py     # 模拟结果分析
├── analyze_multi_source.py   # 模拟与真实数据对比
└── README.md
```

## 环境依赖

建议使用 Python 3.9 及以上版本。

安装依赖：

```bash
pip install numpy pandas matplotlib seaborn openai
```

如果需要运行基于大模型的 thought 关键词抽取，还需要额外配置对应的 API Key。

## 快速开始

### 1. 运行模拟

单次模拟：

```bash
python run_simulation.py --num_runs 1 --rider_num 50 --run_len 360
```

多次模拟：

```bash
python run_simulation.py --num_runs 10 --rider_num 50 --run_len 360 --seed_base 42
```

常用参数：

- `--num_runs`：模拟重复次数，默认 `5`
- `--rider_num`：骑手数量，默认 `50`
- `--run_len`：总步数，默认 `360`
- `--one_day`：一天对应的步数，默认 `120`
- `--order_weight`：订单生成强度系数，默认 `0.3`
- `--seed_base`：随机种子基值，便于复现
- `--no_detail`：只保留聚合结果，不输出每轮详细文件
- `--decision_mode`：骑手决策模式，可选 `auto`、`llm`、`heuristic`
- `--llm_model`：指定 LLM 模型名
- `--llm_base_url`：指定兼容 OpenAI 的模型服务地址

### 2. 启用 LLM 骑手决策

当前版本已经支持“真正基于 LLM 的骑手决策”，包括：

- 每日上下班时间决策
- 每轮候选订单选择
- 决策过程 `thought` 记录

默认推荐先设置环境变量：

```bash
export DASHSCOPE_API_KEY=你的密钥
```

然后运行：

```bash
python run_simulation.py --num_runs 3 --rider_num 20 --run_len 120 --decision_mode llm
```

说明：

- `decision_mode=llm`：强制使用 LLM 决策，若缺少依赖或密钥会直接报错
- `decision_mode=heuristic`：使用规则版骑手
- `decision_mode=auto`：检测到 `DASHSCOPE_API_KEY` 时使用 LLM，否则退回规则版

## 输出结果

运行 `run_simulation.py` 后，结果会保存在 `simulation_results/`：

- `sim_0/`, `sim_1/` ...：每次模拟的单独结果目录
- `rider_summary.csv`：单轮骑手汇总
- `platform_summary.csv`：单轮平台汇总
- `run_config.csv`：单轮模拟配置
- `*_thought.json`：每个骑手的决策 thought 日志
- `aggregated_results.csv`：多轮模拟拼接后的总表
- `summary_statistics.csv`：多轮模拟统计摘要

## 结果分析

### 1. 分析多次模拟结果

```bash
python analyze_simulation.py
```

主要输出：

- 跨轮次骑手指标统计
- 跨轮次平台指标统计
- 稳定性分析
- 柱状图与分布图，保存在 `analysis_results/`

### 2. 与真实实验数据对比

```bash
python analyze_simulation.py --compare
```

或直接执行：

```bash
python analyze_multi_source.py
```

该步骤会将模拟结果与 `exp1` 到 `exp5` 中的真实实验数据进行统一指标对比，并输出对比表与箱线图。

## 论文实验数据分析流程

若只分析已有实验数据，建议执行顺序如下：

1. `python analyze_experiments.py`
2. 如需关键词抽取，先配置 API Key，再运行 `python 1_analyze_thoughts_with_llm.py`
3. `python analyze_thoughts.py`
4. `python analyze_intention_behavior.py`
5. `python 4_generate_experiment_heatmaps.py`
6. `python generate_evolution_sankey.py`
7. `python generate_core_figure.py`

## 指标说明

### 个体层

- `money`：骑手累计收益
- `labor`：骑手累计劳动成本或移动量
- `total_order`：完成订单数
- `stability`：收益稳定性
- `robustness`：对波动的稳健性
- `inv`：收益与成本比值
- `utility`：综合效用

### 平台层

- `profit`：平台收益
- `fairness`：骑手收益分配公平性
- `variety`：骑手策略多样性
- `entropy_increase`：系统演化中的熵增速率
- `utility`：平台综合效用

## 说明

- 本仓库中的模拟代码是对原始 `SocialInvolution` 示例的整理与精简，目标是服务论文复现与公开发布。
- 为保证结构清晰，当前实现保留了论文分析所需的核心机制，但没有引入原工程中的全部复杂依赖与冗余模块。
- 当前仓库同时支持两种骑手决策模式：规则版 `heuristic` 与大模型版 `llm`。
- 在 `llm` 模式下，骑手会输出可追踪的 thought 记录，格式与原始实验中的 `*_thought.json` 保持相近。

