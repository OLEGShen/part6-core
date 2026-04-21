# Part6-Core

This repository contains the core code for the paper's experiments on rider behavior and system evolution in food-delivery platforms. It includes:

- `simulation/` and related execution scripts: a cleaned and publishable simulation derived from the original `SocialInvolution` example
- experiment-analysis scripts for real logs, simulation results, and mechanism-level thought tracing

The current version is organized for reproducibility, repeated simulation, and direct paper-oriented analysis.

## Repository Contents

### 1. Real Experiment Analysis

- `analyze_experiments.py`: computes rider-level and system-level metrics from experimental logs
- `analyze_thoughts.py`: analyzes the relationship between thought text and performance
- `analyze_thoughts_with_llm.py`: extracts thought keywords with an LLM
- `analyze_intention_behavior.py`: analyzes intention-behavior correlations
- `generate_experiment_heatmaps.py`: generates trajectory heatmaps
- `generate_evolution_sankey.py`: generates intention-evolution Sankey diagrams
- `generate_core_figure.py`: generates the combined paper figure

### 2. Refactored Simulation Modules

- `simulation/order.py`: order entity definition
- `simulation/order_generator.py`: Gaussian-mixture order generation
- `simulation/rider.py`: rider agent logic
- `simulation/platform.py`: platform state and system metrics
- `simulation/city.py`: simulation environment coordinator
- `simulation/individual_cal.py`: rider-level metrics
- `simulation/sys_cal.py`: platform-level welfare and involution metrics
- `simulation/dispatch.py`: dispatch and routing heuristics
- `simulation/llm_agent.py`: LLM-backed rider decision module

### 3. Simulation and Causal Analysis

- `run_simulation.py`: runs one or multiple simulations
- `run_observation_analysis.py`: observation analysis corresponding to Algorithm 7-1
- `run_intervention_analysis.py`: intervention analysis corresponding to Algorithm 7-2
- `run_mechanism_analysis.py`: mechanism analysis corresponding to Algorithm 7-3
- `analyze_simulation.py`: analyzes repeated simulation outputs
- `analyze_multi_source.py`: compares simulation results with `exp1` to `exp5`

## Directory Overview

```text
part6-core/
├── simulation/                 # Refactored simulation code
├── simulation_results/         # Simulation outputs
├── analysis_results/           # Figures and summary tables
├── paper_figures/              # Paper figure assets and source files
├── config.py                   # Centralized paper-aligned configuration
├── run_simulation.py           # Simulation entrypoint
├── run_observation_analysis.py # Algorithm 7-1
├── run_intervention_analysis.py# Algorithm 7-2
├── run_mechanism_analysis.py   # Algorithm 7-3
└── README.md
```

## Requirements

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want to run LLM-based thought extraction or LLM rider decisions, you also need to configure an API key.

## Quick Start

The default parameters in `python run_simulation.py` are already aligned with Section 7.4.2 of the paper:

- `rider_num = 100`
- `run_len = 3600`
- `num_runs = 10`
- `business_district_num = 10`
- `llm temperature = 0`
- `llm model = DeepSeek-R1-Distill-Qwen-32B`

### 1. Run the simulation

Default paper-aligned run:

```bash
python run_simulation.py
```

Single-run test:

```bash
python run_simulation.py --num_runs 1
```

Common parameters:

- `--num_runs`: number of repeated simulations, default `10`
- `--rider_num`: number of riders, default `100`
- `--run_len`: total simulation steps, default `3600`
- `--one_day`: steps per day, default `120`
- `--order_weight`: order-generation intensity, default `0.3`
- `--seed_base`: base random seed, default `42`
- `--no_detail`: skip per-run detailed outputs
- `--decision_mode`: `auto`, `llm`, or `heuristic`
- `--llm_model`: LLM model name
- `--llm_base_url`: OpenAI-compatible API endpoint

### 2. Enable LLM Rider Decisions

The current version supports real LLM-based rider decisions for:

- daily work-time adjustment
- order-selection decisions
- traceable `*_thought.json` outputs

Set the API key first:

```bash
export DASHSCOPE_API_KEY=your_key
```

Then run:

```bash
python run_simulation.py --decision_mode llm
```

Notes:

- `decision_mode=llm`: forces LLM decisions and fails fast if dependencies or keys are missing
- `decision_mode=heuristic`: uses the rule-based rider
- `decision_mode=auto`: uses LLM if `DASHSCOPE_API_KEY` is available, otherwise falls back to the heuristic version

## Outputs

After running `run_simulation.py`, results are stored in `simulation_results/`:

- `sim_0/`, `sim_1/`, ...: one folder per run
- `rider_summary.csv`: rider-level summary for one run
- `platform_summary.csv`: platform-level summary for one run
- `run_config.csv`: configuration of one run
- `time_series.csv`: time-series metrics for one run
- `rider_positions.json`: rider position snapshots for heatmap analysis
- `*_thought.json`: rider decision traces
- `aggregated_results.csv`: combined multi-run results
- `summary_statistics.csv`: aggregated summary statistics

## Analysis Pipeline

### 1. Simulation-Level Analysis

```bash
python analyze_simulation.py
```

### 2. Observation Layer

```bash
python run_observation_analysis.py
```

This generates:

- Figure 7-9(a): involution-level distribution
- Figure 7-9(b): involution timeline across repeated runs
- Figure 7-10: rider activity heatmaps

### 3. Intervention Layer

```bash
python run_intervention_analysis.py
```

This generates:

- Figure 7-11(a): intervention boxplot
- Figure 7-11(d): SEM path coefficients

### 4. Mechanism Layer

```bash
python run_mechanism_analysis.py
```

This generates:

- Figure 7-12(a-c): clustered intention evolution
- Figure 7-12(d-f): intention-behavior correlation heatmaps
- Figure 7-12(g): intention-evolution Sankey diagram
- Figure 7-12(h-k): behavior transition comparison under different contexts

### 5. One-Click Reproduction

```bash
bash reproduce_paper.sh
```

## Paper Mapping

### Figure/Table Mapping

| Paper Figure/Table | Content | Script | Key Function |
|----------|------|---------|---------|
| Table 7-3 | Simulation vs. real data comparison (MAE/RMSE/r) | `analyze_simulation.py` | `compare_with_zomato()` |
| Table 7-4 | Real-computational system mapping | documentation only | — |
| Figure 7-9(a) | Involution-level experimental distribution | `run_observation_analysis.py` | `plot_involution_distribution()` |
| Figure 7-9(b) | Involution timeline | `run_observation_analysis.py` | `plot_involution_timeline()` |
| Figure 7-10 | Rider activity heatmap | `generate_experiment_heatmaps.py` | `generate_heatmap()` |
| Figure 7-11(a) | Intervention boxplot | `run_intervention_analysis.py` | `plot_intervention_boxplot()` |
| Figure 7-11(d) | SEM path coefficients | `run_intervention_analysis.py` | `compute_sem_coefficients()` |
| Figure 7-12(a-c) | Intention clustering evolution | `run_mechanism_analysis.py` | `cluster_intentions()` |
| Figure 7-12(d-f) | Intention-behavior correlation | `analyze_intention_behavior.py` | `compute_correlation_matrix()` |
| Figure 7-12(g) | Intention evolution Sankey diagram | `generate_evolution_sankey.py` | `build_sankey()` |

### Formula Mapping

| Paper Formula | Meaning | File | Function |
|------------|------|---------|-------|
| Rider Utility CRRA | Individual utility | `simulation/individual_cal.py` | `compute_utility()` |
| Swf social welfare | Platform welfare | `simulation/sys_cal.py` | `compute_swf()` |
| Involution(t) | Involution index | `simulation/sys_cal.py` | `compute_involution()` |
| Formula (1) `I(X;Y\|Z)` | Conditional mutual information | `analyze_simulation.py` | `conditional_mutual_info()` |
| Formula (2) `Effect(x)` | Intervention effect ATE | `run_intervention_analysis.py` | `compute_ate()` |
| Formula (3) `P(Y\|do(X))` | Backdoor adjustment | `run_intervention_analysis.py` | `backdoor_adjustment()` |
| Mechanism Formula (1) `Cs/Cr` | Dual-view thought extraction | `run_mechanism_analysis.py` | `extract_dual_thoughts()` |
| Mechanism Formula (2) | Emergent intention detection | `run_mechanism_analysis.py` | `detect_emergent_intention()` |
| Mechanism Formula (3) | Cosine similarity | `run_mechanism_analysis.py` | `cosine_similarity()` |
| Mechanism Formula (4) | Intention clustering | `run_mechanism_analysis.py` | `cluster_intentions()` |

## Notes

- This repository is a cleaned and compressed version of the original `SocialInvolution` project for paper reproduction and public release.
- The code now supports both a rule-based rider (`heuristic`) and an LLM-based rider (`llm`).
- In `llm` mode, the rider outputs traceable thought records in `*_thought.json`.
- The analysis scripts are designed to work even if `exp1` to `exp5` are not included in the public repository, as long as equivalent input data are provided later.
