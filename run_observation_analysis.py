"""Observation analysis pipeline corresponding to Algorithm 7-1."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

from config import ANALYSIS_RESULTS_DIR, SIMULATION_CONFIG, THRESHOLD_CONFIG
from run_simulation import run_single_simulation


OUTPUT_DIR = ANALYSIS_RESULTS_DIR


def classify_involution_level(value):
    """Classify involution levels using the paper thresholds."""

    if value <= THRESHOLD_CONFIG.low_involution:
        return "Low"
    if value <= THRESHOLD_CONFIG.high_involution:
        return "Medium"
    return "High"


def run_observation_experiments(
    num_runs=SIMULATION_CONFIG.num_runs,
    rider_num=SIMULATION_CONFIG.rider_num,
    run_len=SIMULATION_CONFIG.run_len,
):
    """Run repeated experiments with identical parameters and different seeds."""

    records = []
    run_results = []
    for run_id in range(num_runs):
        seed = SIMULATION_CONFIG.seed_base + run_id
        results, _ = run_single_simulation(
            run_id=run_id,
            rider_num=rider_num,
            run_len=run_len,
            one_day=SIMULATION_CONFIG.one_day,
            order_weight=SIMULATION_CONFIG.order_weight,
            seed=seed,
            save_detail=True,
            decision_mode="heuristic",
        )
        time_series = pd.DataFrame(results["time_series"])
        final_involution = float(time_series["involution"].iloc[-1]) if not time_series.empty else 0.0
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "final_involution": final_involution,
                "mean_involution": time_series["involution"].mean(),
                "max_involution": time_series["involution"].max(),
                "mean_active_riders": time_series["active_riders"].mean(),
                "mean_swf": time_series["swf"].mean(),
                "mean_profit": time_series["platform_profit"].mean(),
                "involution_level": classify_involution_level(final_involution),
            }
        )
        run_results.append((run_id, results))
    return pd.DataFrame(records), run_results


def plot_involution_distribution(summary_df, output_dir=OUTPUT_DIR):
    """Plot Figure 7-9(a): experimental distribution of involution levels."""

    counts = summary_df["involution_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#8dd3c7", "#ffffb3", "#fb8072"]
    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_title("Figure 7-9(a) Involution Level Distribution")
    axes[0].set_ylabel("Experiment Count")

    axes[1].pie(counts.values, labels=counts.index, autopct="%1.0f%%", colors=colors)
    axes[1].set_title("Figure 7-9(a) Pie Summary")

    fig.tight_layout()
    output_path = Path(output_dir) / "fig7-09_observation_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_involution_timeline(run_results, output_dir=OUTPUT_DIR):
    """Plot Figure 7-9(b): involution trajectories over time."""

    fig, ax = plt.subplots(figsize=(12, 6))
    max_len = 0
    stacked = []
    for run_id, results in run_results:
        time_series = pd.DataFrame(results["time_series"])
        max_len = max(max_len, len(time_series))
        stacked.append(np.asarray(time_series["involution"].values))
        ax.plot(time_series["step"], time_series["involution"], alpha=0.35, linewidth=1.2, label=f"Run {run_id}")

    matrix = np.full((len(stacked), max_len), np.nan)
    for index, row in enumerate(stacked):
        matrix[index, : len(row)] = row
    mean_curve = np.nanmean(matrix, axis=0)
    ax.plot(np.arange(max_len), mean_curve, color="black", linewidth=2.5, label="Mean")
    ax.axhline(THRESHOLD_CONFIG.low_involution, linestyle="--", color="#4daf4a", linewidth=1)
    ax.axhline(THRESHOLD_CONFIG.high_involution, linestyle="--", color="#e41a1c", linewidth=1)
    ax.set_title("Figure 7-9(b) Involution Dynamics Across Experiments")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Involution Index")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    output_path = Path(output_dir) / "fig7-09_observation_timeline.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_heatmap(run_results, output_dir=OUTPUT_DIR):
    """Plot Figure 7-10: rider activity heatmaps at multiple time slices."""

    if not run_results:
        return None

    selected_steps = [0.25, 0.5, 0.75, 1.0]
    sample_time_series = pd.DataFrame(run_results[0][1]["time_series"])
    total_steps = len(sample_time_series)
    cutoffs = [min(total_steps - 1, max(0, math.floor(total_steps * ratio) - 1)) for ratio in selected_steps]

    fig, axes = plt.subplots(1, len(cutoffs), figsize=(4 * len(cutoffs), 4))
    if len(cutoffs) == 1:
        axes = [axes]

    for axis, cutoff in zip(axes, cutoffs):
        points = []
        for _, results in run_results:
            time_series = pd.DataFrame(results["time_series"])
            history = time_series["rider_positions"].iloc[cutoff]
            points.extend([(item["x"], item["y"]) for item in history])
        df = pd.DataFrame(points, columns=["x", "y"])
        heatmap, xedges, yedges = np.histogram2d(df["x"], df["y"], bins=10, range=[[0, 100], [0, 100]])
        sns.heatmap(heatmap.T, ax=axis, cmap="YlOrRd", cbar=False, square=True)
        axis.set_title(f"Step {cutoff + 1}")
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle("Figure 7-10 Rider Activity Heatmaps", y=0.98)
    fig.tight_layout()
    output_path = Path(output_dir) / "fig7-10_activity_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def cluster_evolution_results(summary_df, output_dir=OUTPUT_DIR):
    """Cluster repeated experiments and output the key influencing factors."""

    features = summary_df[["final_involution", "mean_active_riders", "mean_swf", "mean_profit"]].fillna(0.0)
    kmeans = KMeans(n_clusters=min(3, len(features)), random_state=SIMULATION_CONFIG.seed_base, n_init=10)
    labels = kmeans.fit_predict(features)
    summary_df = summary_df.copy()
    summary_df["cluster"] = labels

    corr_target = summary_df[["final_involution", "mean_active_riders", "mean_swf", "mean_profit"]].corr()["final_involution"]
    key_factors = corr_target.drop(labels=["final_involution"]).abs().sort_values(ascending=False)

    summary_path = Path(output_dir) / "observation_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    factor_path = Path(output_dir) / "observation_key_factors.txt"
    with open(factor_path, "w", encoding="utf-8") as file:
        file.write("Key influencing factors from Algorithm 7-1 clustering\n")
        for factor_name, score in key_factors.items():
            file.write(f"{factor_name}: {score:.4f}\n")
    return summary_df, list(key_factors.index)


def main():
    parser = argparse.ArgumentParser(description="Run observation analysis (Algorithm 7-1).")
    parser.add_argument("--num_runs", type=int, default=SIMULATION_CONFIG.num_runs)
    parser.add_argument("--rider_num", type=int, default=SIMULATION_CONFIG.rider_num)
    parser.add_argument("--run_len", type=int, default=SIMULATION_CONFIG.run_len)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_df, run_results = run_observation_experiments(
        num_runs=args.num_runs,
        rider_num=args.rider_num,
        run_len=args.run_len,
    )
    plot_involution_distribution(summary_df)
    plot_involution_timeline(run_results)
    generate_heatmap(run_results)
    cluster_evolution_results(summary_df)


if __name__ == "__main__":
    main()
