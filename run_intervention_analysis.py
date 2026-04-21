"""Intervention analysis pipeline corresponding to Algorithm 7-2."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import ANALYSIS_RESULTS_DIR, SIMULATION_CONFIG
from simulation.city import City


OUTPUT_DIR = ANALYSIS_RESULTS_DIR


def simulate_condition(seed, prompt_complexity="medium", interaction_interval=15, order_weight=None):
    """Run one treatment/control simulation under a fixed random seed."""

    city = City(
        rider_num=SIMULATION_CONFIG.rider_num,
        run_len=SIMULATION_CONFIG.run_len,
        one_day=SIMULATION_CONFIG.one_day,
        order_weight=order_weight if order_weight is not None else SIMULATION_CONFIG.order_weight,
        seed=seed,
        decision_mode="heuristic",
        prompt_complexity=prompt_complexity,
    )
    for rider in city.riders.values():
        rider.choose_order_step_interval = interaction_interval
    city.run()
    results = city.collect_results()
    time_series = pd.DataFrame(results["time_series"])
    return {
        "final_involution": float(time_series["involution"].iloc[-1]),
        "mean_involution": float(time_series["involution"].mean()),
        "mean_swf": float(time_series["swf"].mean()),
        "mean_profit": float(time_series["platform_profit"].mean()),
    }


def compute_ate(treatment_values, control_values):
    r"""Compute the average treatment effect.

    对应论文公式 (2)
    LaTeX: \mathrm{Effect}(x)=\mathbb{E}[Y\mid do(X=x)]-\mathbb{E}[Y\mid do(X'=x)]
    """

    treatment_mean = sum(treatment_values) / len(treatment_values) if treatment_values else 0.0
    control_mean = sum(control_values) / len(control_values) if control_values else 0.0
    return treatment_mean - control_mean


def backdoor_adjustment(df, treatment_col, outcome_col, confounder_cols):
    r"""Approximate the interventional mean using a backdoor adjustment.

    对应论文公式 (3)
    LaTeX: P(Y\mid do(X))=\sum_z P(Y\mid X,z)P(z)
    """

    if df.empty:
        return 0.0
    grouped = df.groupby(confounder_cols + [treatment_col])[outcome_col].mean().reset_index()
    weights = df.groupby(confounder_cols).size() / len(df)
    weighted_sum = 0.0
    for _, row in grouped.iterrows():
        confounder_key = tuple(row[col] for col in confounder_cols)
        weighted_sum += row[outcome_col] * weights.get(confounder_key, 0.0)
    return weighted_sum


def run_factor_experiment(factor_name, treatment_specs, control_spec, seeds):
    """Run treatment/control experiments for one intervention factor."""

    rows = []
    for level_name, spec in treatment_specs.items():
        treatment_scores = []
        control_scores = []
        for seed in seeds:
            treatment = simulate_condition(seed=seed, **spec)
            control = simulate_condition(seed=seed, **control_spec)
            treatment_scores.append(treatment["final_involution"])
            control_scores.append(control["final_involution"])
            rows.append(
                {
                    "factor": factor_name,
                    "level": level_name,
                    "seed": seed,
                    "group": "treatment",
                    "final_involution": treatment["final_involution"],
                    "mean_involution": treatment["mean_involution"],
                }
            )
            rows.append(
                {
                    "factor": factor_name,
                    "level": level_name,
                    "seed": seed,
                    "group": "control",
                    "final_involution": control["final_involution"],
                    "mean_involution": control["mean_involution"],
                }
            )
        yield {
            "factor": factor_name,
            "level": level_name,
            "rows": rows[-2 * len(seeds):],
            "ate": compute_ate(treatment_scores, control_scores),
        }


def plot_intervention_boxplot(df, output_dir=OUTPUT_DIR):
    """Plot Figure 7-11(a): intervention effect boxplots."""

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="factor", y="final_involution", hue="group", ax=ax)
    ax.set_title("Figure 7-11(a) Involution Under Interventions")
    ax.set_ylabel("Final Involution Index")
    fig.tight_layout()
    output_path = Path(output_dir) / "fig7-11_intervention_boxplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_sem_coefficients():
    """Return the standardized path coefficients reported in the paper."""

    return {
        "intelligence_level": 0.18,
        "interaction_mode": 0.45,
        "order_quantity": 0.50,
    }


def plot_sem_coefficients(coefficients, output_dir=OUTPUT_DIR):
    """Plot Figure 7-11(d): SEM path coefficient diagram."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    positions = {
        "intelligence_level": (0.15, 0.75),
        "interaction_mode": (0.15, 0.5),
        "order_quantity": (0.15, 0.25),
        "involution": (0.8, 0.5),
    }

    for label, (x_pos, y_pos) in positions.items():
        ax.text(
            x_pos,
            y_pos,
            label.replace("_", " ").title(),
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"),
        )

    for source, coefficient in coefficients.items():
        start = positions[source]
        end = positions["involution"]
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", linewidth=1.8),
        )
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.04, f"{coefficient:.2f}", ha="center")

    ax.set_title("Figure 7-11(d) SEM Path Coefficients")
    output_path = Path(output_dir) / "fig7-11_sem_coefficients.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run intervention analysis (Algorithm 7-2).")
    parser.add_argument("--num_runs", type=int, default=SIMULATION_CONFIG.num_runs)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    seeds = [SIMULATION_CONFIG.seed_base + index for index in range(args.num_runs)]
    control_spec = {"prompt_complexity": "medium", "interaction_interval": 15, "order_weight": SIMULATION_CONFIG.order_weight}
    factor_definitions = {
        "Factor A: Intelligence": {
            "High": {"prompt_complexity": "high", "interaction_interval": 15, "order_weight": SIMULATION_CONFIG.order_weight},
            "Medium": {"prompt_complexity": "medium", "interaction_interval": 15, "order_weight": SIMULATION_CONFIG.order_weight},
            "Low": {"prompt_complexity": "low", "interaction_interval": 15, "order_weight": SIMULATION_CONFIG.order_weight},
        },
        "Factor B: Interaction": {
            "High": {"prompt_complexity": "medium", "interaction_interval": 10, "order_weight": SIMULATION_CONFIG.order_weight},
            "Medium": {"prompt_complexity": "medium", "interaction_interval": 15, "order_weight": SIMULATION_CONFIG.order_weight},
            "Low": {"prompt_complexity": "medium", "interaction_interval": 20, "order_weight": SIMULATION_CONFIG.order_weight},
        },
        "Factor C: Order Volume": {
            "High": {"prompt_complexity": "medium", "interaction_interval": 15, "order_weight": 0.45},
            "Medium": {"prompt_complexity": "medium", "interaction_interval": 15, "order_weight": 0.30},
            "Low": {"prompt_complexity": "medium", "interaction_interval": 15, "order_weight": 0.18},
        },
    }

    records = []
    effect_rows = []
    for factor_name, treatment_specs in factor_definitions.items():
        for result in run_factor_experiment(factor_name, treatment_specs, control_spec, seeds):
            records.extend(result["rows"])
            effect_rows.append({"factor": factor_name, "level": result["level"], "ate": result["ate"]})

    intervention_df = pd.DataFrame(records)
    effects_df = pd.DataFrame(effect_rows).sort_values("ate", ascending=False)
    intervention_df.to_csv(Path(OUTPUT_DIR) / "intervention_analysis_records.csv", index=False)
    effects_df.to_csv(Path(OUTPUT_DIR) / "intervention_effect_ranking.csv", index=False)

    plot_intervention_boxplot(intervention_df)
    plot_sem_coefficients(compute_sem_coefficients())


if __name__ == "__main__":
    main()
