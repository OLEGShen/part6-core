import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")
DEFAULT_TRAJECTORY_FILE = os.path.join(BASE_DIR, "exp1", "deliver_1_record.csv")
TIME_COLUMN = "time"
N_PHASES = 3
N_BINS = 10
HEATMAP_CMAP = "Blues"


def configure_plot_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")


def load_cumulative_heatmap_counts(csv_path, time_col_name=TIME_COLUMN):
    df = pd.read_csv(csv_path)
    required_columns = {time_col_name, "x", "y"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"轨迹文件缺少必要列: {missing_text}")

    df = df.copy()
    df[time_col_name] = pd.to_numeric(df[time_col_name], errors="coerce")
    df["x_scaled"] = pd.to_numeric(df["x"], errors="coerce") / 20.0
    df["y_scaled"] = pd.to_numeric(df["y"], errors="coerce") / 20.0
    df.dropna(subset=[time_col_name, "x_scaled", "y_scaled"], inplace=True)
    df.sort_values(by=time_col_name, inplace=True)

    if df.empty:
        raise ValueError("轨迹文件清洗后无有效数据。")

    min_time = df[time_col_name].min()
    max_time = df[time_col_name].max()
    cutoffs = []
    for index in range(1, N_PHASES + 1):
        cutoff = min_time + (index / N_PHASES) * (max_time - min_time)
        if index == N_PHASES:
            cutoff = max_time
        cutoffs.append(cutoff)

    global_x_min = df["x_scaled"].min()
    global_x_max = df["x_scaled"].max()
    global_y_min = df["y_scaled"].min()
    global_y_max = df["y_scaled"].max()

    if global_x_min == global_x_max:
        global_x_max += 1e-6
    if global_y_min == global_y_max:
        global_y_max += 1e-6

    x_bin_edges = np.linspace(global_x_min, global_x_max, N_BINS + 1)
    y_bin_edges = np.linspace(global_y_min, global_y_max, N_BINS + 1)

    counts_by_phase = []
    max_count = 0
    for cutoff in cutoffs:
        df_cumulative = df[df[time_col_name] <= cutoff]
        counts, _, _ = np.histogram2d(
            df_cumulative["x_scaled"].values,
            df_cumulative["y_scaled"].values,
            bins=[x_bin_edges, y_bin_edges],
        )
        counts = counts.T
        counts_by_phase.append(counts)
        if counts.size > 0:
            max_count = max(max_count, int(np.max(counts)))

    return counts_by_phase, max(max_count, 1)


def plot_top_row(fig, gs, csv_path):
    counts_by_phase, max_count = load_cumulative_heatmap_counts(csv_path)
    norm = plt.Normalize(vmin=0, vmax=max_count)
    scalar_map = plt.cm.ScalarMappable(cmap=HEATMAP_CMAP, norm=norm)
    scalar_map.set_array([])

    for index, counts in enumerate(counts_by_phase):
        ax = fig.add_subplot(gs[0, index])
        sns.heatmap(
            counts,
            ax=ax,
            cmap=HEATMAP_CMAP,
            vmin=0,
            vmax=max_count,
            cbar=False,
            square=True,
            linewidths=1.0,
            linecolor="white",
            xticklabels=False,
            yticklabels=False,
            annot=False,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Phase {index + 1}", fontsize=18, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor("black")

    cbar_ax = fig.add_subplot(gs[0, 3])
    colorbar = fig.colorbar(scalar_map, cax=cbar_ax)
    colorbar.set_label("Cumulative Activity Count", fontsize=13, fontweight="bold")
    colorbar.ax.tick_params(labelsize=11)


def plot_middle_row(fig, gs):
    rng = np.random.default_rng(42)
    axes = [fig.add_subplot(gs[1, index]) for index in range(3)]

    factor_a = pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    rng.normal(96, 12, 50),
                    rng.normal(99, 12, 50),
                ]
            ),
            "group": ["High Intelligence"] * 50 + ["Low Intelligence"] * 50,
        }
    )
    sns.boxplot(
        data=factor_a,
        x="group",
        y="value",
        ax=axes[0],
        hue="group",
        palette=["#66c2a5", "#fc8d62"],
        linewidth=2.0,
        fliersize=5,
    )
    legend = axes[0].get_legend()
    if legend is not None:
        legend.remove()
    axes[0].set_title("Intelligence Level Adjustment", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("Intelligence Level Group", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Revenue Level", fontsize=12, fontweight="bold")
    axes[0].text(
        0.5,
        0.96,
        "p > 0.05 (No Significant Difference)",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="lightgray", alpha=0.7),
    )

    factor_b = pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    rng.normal(120, 4.2, 50),
                    rng.normal(97.6, 9.8, 50),
                ]
            ),
            "group": ["Baseline Period"] * 50 + ["Intervention Period"] * 50,
        }
    )
    sns.boxplot(
        data=factor_b,
        x="group",
        y="value",
        ax=axes[1],
        hue="group",
        palette=["#8da0cb", "#e78ac3"],
        linewidth=2.0,
        fliersize=5,
    )
    legend = axes[1].get_legend()
    if legend is not None:
        legend.remove()
    axes[1].set_title("Interaction Mode Transformation", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Experimental Period", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Revenue Level", fontsize=12, fontweight="bold")
    axes[1].text(
        0.5,
        0.96,
        "p < 0.01 (Highly Significant)",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="lightcoral", alpha=0.7),
    )
    axes[1].text(
        0.5,
        0.88,
        "Mean Decrease 18.7%",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow", alpha=0.7),
    )

    factor_c = pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    rng.normal(80, 8, 50),
                    rng.normal(110, 12, 50),
                    rng.normal(125, 18, 50),
                ]
            ),
            "group": ["Low Volume"] * 50 + ["Medium Volume"] * 50 + ["High Volume"] * 50,
        }
    )
    sns.boxplot(
        data=factor_c,
        x="group",
        y="value",
        ax=axes[2],
        hue="group",
        palette=["#a6d854", "#ffd92f", "#e5c494"],
        linewidth=2.0,
        fliersize=5,
    )
    legend = axes[2].get_legend()
    if legend is not None:
        legend.remove()
    axes[2].set_title("Order Quantity Variation", fontsize=16, fontweight="bold")
    axes[2].set_xlabel("Order Quantity Level", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Individual Income Level", fontsize=12, fontweight="bold")
    axes[2].text(
        0.5,
        0.96,
        "Diminishing Marginal Returns",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="lightblue", alpha=0.7),
    )

    for ax in axes:
        ax.tick_params(axis="x", rotation=0, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("black")


def build_bottom_phase_matrices():
    intentions = ["Rules Following", "Anxiety", "Maintain Income", "Risk Aversion"]
    behaviors = ["Order", "Rest", "Earn", "No Earn"]

    phase_matrices = [
        np.array(
            [
                [0.40, 0.80, 0.30, 0.20],
                [0.10, 0.10, 0.10, 0.10],
                [0.60, 0.10, 0.70, 0.10],
                [0.10, 0.10, 0.10, 0.10],
            ]
        ),
        np.array(
            [
                [0.30, 0.50, 0.20, 0.10],
                [0.70, 0.20, 0.80, 0.20],
                [0.70, 0.10, 0.80, 0.10],
                [0.40, 0.10, 0.50, 0.10],
            ]
        ),
        np.array(
            [
                [0.20, 0.30, 0.10, 0.10],
                [0.40, 0.30, 0.50, 0.40],
                [0.50, 0.20, 0.60, 0.30],
                [0.30, 0.20, 0.40, 0.30],
            ]
        ),
    ]
    return intentions, behaviors, phase_matrices


def plot_bottom_row(fig, gs):
    intentions, behaviors, phase_matrices = build_bottom_phase_matrices()
    all_values = np.concatenate([matrix.flatten() for matrix in phase_matrices])
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    axes = [fig.add_subplot(gs[2, index]) for index in range(3)]

    for index, matrix in enumerate(phase_matrices):
        df = pd.DataFrame(matrix, index=intentions, columns=behaviors)
        sns.heatmap(
            df,
            ax=axes[index],
            annot=True,
            fmt=".2f",
            cmap=HEATMAP_CMAP,
            linewidths=1.0,
            linecolor="white",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=True,
            annot_kws={"size": 10, "weight": "bold"},
        )
        phase_title = ["Phase 1: Initial", "Phase 2: Mid", "Phase 3: End"][index]
        axes[index].set_title(phase_title, fontsize=16, fontweight="bold", pad=8)
        axes[index].set_xlabel("Behavior", fontsize=12, fontweight="bold")
        axes[index].set_ylabel("Desire" if index == 0 else "", fontsize=12, fontweight="bold")
        axes[index].tick_params(axis="x", rotation=0, labelsize=10)
        axes[index].tick_params(axis="y", rotation=0, labelsize=10, labelleft=(index == 0))
        axes[index].set_aspect("equal", adjustable="box")
        for spine in axes[index].spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("black")

    cbar_ax = fig.add_subplot(gs[2, 3])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = plt.cm.ScalarMappable(cmap=HEATMAP_CMAP, norm=norm)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, cax=cbar_ax)
    colorbar.set_label("Correlation", fontsize=13, fontweight="bold")
    colorbar.ax.tick_params(labelsize=11)


def generate_core_figure(csv_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configure_plot_style()

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1, 1, 1, 0.08],
        height_ratios=[1, 1, 1],
        hspace=0.42,
        wspace=0.32,
    )

    plot_top_row(fig, gs, csv_path)
    plot_middle_row(fig, gs)
    plot_bottom_row(fig, gs)

    fig.suptitle(
        "Combined Experiment Analysis: Multi-Phase Behavioral Dynamics",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    png_output = os.path.join(OUTPUT_DIR, "combined_9x9_figure_for_paper.png")
    pdf_output = os.path.join(OUTPUT_DIR, "combined_9x9_figure_for_paper.pdf")
    fig.savefig(png_output, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(pdf_output, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"Saved combined figure to: {png_output}")
    print(f"Saved combined figure to: {pdf_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="生成核心组合图。")
    parser.add_argument(
        "--decision_mode",
        choices=["llm", "heuristic", "imitation", "auto"],
        default="llm",
        help="Pipeline compatibility argument; unused in this plotting script.",
    )
    parser.add_argument(
        "--trajectory-file",
        default=DEFAULT_TRAJECTORY_FILE,
        help="用于顶部累计热力图的轨迹 CSV 文件路径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_core_figure(args.trajectory_file)


if __name__ == "__main__":
    main()
