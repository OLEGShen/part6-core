# 核心上传包

## 桑基图说明

- `generate_evolution_sankey.py`：基于 `exp1` 到 `exp5` 的真实工时决策记录生成演化过程桑基图
- 核心意图缩减为三类：`稳健维持`、`竞争扩张`、`风险收缩`
- 意图识别结合 `thought` 文本、上下班时间调整和排名变化进行联合判断
- 输出结果：`analysis_results/evolution_process_sankey.html`

## 目录内容

- `exp1` 到 `exp5`：五组实验原始数据
- `analyze_experiments.py`：统计骑手与系统级指标
- `analyze_thoughts.py`：分析 thought 文本与绩效关系
- `1_analyze_thoughts_with_llm.py`：用大模型抽取 thought 关键词
- `analyze_intention_behavior.py`：联合分析关键词与行为变化
- `4_generate_experiment_heatmaps.py`：生成基于真实轨迹的热力图
- `generate_evolution_sankey.py`：基于真实 thought 数据生成演化过程桑基图

## 建议执行顺序

1. 运行 `analyze_experiments.py`
2. 如需 LLM 关键词结果，先设置环境变量 `DASHSCOPE_API_KEY`，再运行 `1_analyze_thoughts_with_llm.py`
3. 运行 `analyze_thoughts.py`
4. 运行 `analyze_intention_behavior.py`
5. 如需轨迹可视化，运行 `4_generate_experiment_heatmaps.py`
6. 如需演化过程桑基图，运行 `generate_evolution_sankey.py`

<br />
