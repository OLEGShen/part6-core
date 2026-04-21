import os
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import math


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_DIR = os.path.join(BASE_DIR, 'simulation_results')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis_results')


def calculate_distance(x1, y1, x2, y2):
    try:
        return math.sqrt((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2)
    except (ValueError, TypeError):
        return 0.0


def conditional_mutual_info(x, y, z, bins=10):
    r"""Estimate conditional mutual information I(X;Y|Z) with histogram binning.

    对应论文公式 (1)
    LaTeX: I(X;Y\mid Z)
    """

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        return 0.0

    x_bins = pd.qcut(x, q=min(bins, len(np.unique(x))), duplicates="drop", labels=False)
    y_bins = pd.qcut(y, q=min(bins, len(np.unique(y))), duplicates="drop", labels=False)
    z_bins = pd.qcut(z, q=min(bins, len(np.unique(z))), duplicates="drop", labels=False)
    df = pd.DataFrame({"x": x_bins, "y": y_bins, "z": z_bins}).dropna()
    if df.empty:
        return 0.0

    total = len(df)
    mi = 0.0
    for z_value, group in df.groupby("z"):
        p_z = len(group) / total
        p_xy = group.groupby(["x", "y"]).size() / len(group)
        p_x = group.groupby("x").size() / len(group)
        p_y = group.groupby("y").size() / len(group)
        for (x_value, y_value), p_xy_value in p_xy.items():
            denom = p_x[x_value] * p_y[y_value]
            if p_xy_value > 0 and denom > 0:
                mi += p_z * p_xy_value * np.log(p_xy_value / denom)
    return float(mi)


def analyze_simulation_results():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    aggregated_file = os.path.join(SIMULATION_DIR, 'aggregated_results.csv')
    summary_file = os.path.join(SIMULATION_DIR, 'summary_statistics.csv')

    if not os.path.exists(aggregated_file):
        print(f"Error: Aggregated results not found at {aggregated_file}")
        print("Please run run_simulation.py first.")
        return

    df = pd.read_csv(aggregated_file)

    print(f"Loaded {len(df)} records from {len(df['run_id'].unique())} simulation runs")

    _analyze_rider_metrics(df)
    _analyze_platform_metrics(df)
    _analyze_multi_run_statistics(df)
    _generate_visualizations(df)


def _analyze_rider_metrics(df):
    print("\n" + "=" * 60)
    print("Rider Metrics Analysis (Averaged across runs)")
    print("=" * 60)

    rider_metrics = ['money', 'labor', 'total_order', 'stability', 'robustness', 'inv', 'utility']
    rider_stats = df.groupby('rider_id')[rider_metrics].mean()

    print("\nOverall Rider Statistics:")
    for metric in rider_metrics:
        mean_val = rider_stats[metric].mean()
        std_val = rider_stats[metric].std()
        print(f"  {metric}: mean={mean_val:.4f}, std={std_val:.4f}")

    return rider_stats


def _analyze_platform_metrics(df):
    print("\n" + "=" * 60)
    print("Platform Metrics Analysis (Averaged across runs)")
    print("=" * 60)

    platform_metrics = ['platform_profit', 'platform_fairness', 'platform_variety',
                       'platform_entropy', 'platform_utility']

    for metric in platform_metrics:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        print(f"  {metric}: mean={mean_val:.4f}, std={std_val:.4f}")

    return df[platform_metrics].describe()


def _analyze_multi_run_statistics(df):
    print("\n" + "=" * 60)
    print("Multi-Run Stability Analysis")
    print("=" * 60)

    run_summary = df.groupby('run_id').agg({
        'money': 'mean',
        'utility': 'mean',
        'platform_utility': 'first',
        'platform_profit': 'first'
    })

    print("\nCross-run summary:")
    print(run_summary)

    print("\nStability metrics (lower std = more stable):")
    for col in ['money', 'utility', 'platform_utility', 'platform_profit']:
        std_val = run_summary[col].std()
        cv = std_val / run_summary[col].mean() if run_summary[col].mean() != 0 else 0
        print(f"  {col}: std={std_val:.4f}, CV={cv:.4f}")

    return run_summary


def _generate_visualizations(df):
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    rider_metrics = ['money', 'labor', 'total_order', 'stability', 'robustness', 'inv', 'utility']

    for metric in rider_metrics:
        plt.figure(figsize=(10, 6))
        run_means = df.groupby('run_id')[metric].mean()
        plt.bar(range(len(run_means)), run_means.values)
        plt.xlabel('Simulation Run')
        plt.ylabel(metric.capitalize())
        plt.title(f'Rider {metric.capitalize()} Across Simulation Runs')
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f'rider_{metric}_comparison.png'))
        plt.close()

    plt.figure(figsize=(12, 6))
    platform_metrics = ['platform_profit', 'platform_fairness', 'platform_variety',
                       'platform_entropy', 'platform_utility']
    run_platform = df.groupby('run_id')[platform_metrics].first()
    run_platform.plot(kind='bar', ax=plt.gca())
    plt.xlabel('Simulation Run')
    plt.ylabel('Value')
    plt.title('Platform Metrics Across Simulation Runs')
    plt.legend(loc='upper right', labels=[m.replace('platform_', '') for m in platform_metrics])
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'platform_metrics_comparison.png'))
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, metric in enumerate(rider_metrics):
        if i < len(axes):
            run_means = df.groupby('run_id')[metric].mean()
            if len(run_means.values) > 0:
                axes[i].hist(run_means.values, bins=min(10, len(run_means.values)), alpha=0.7)
            axes[i].set_title(metric.capitalize())
            axes[i].grid(axis='y', alpha=0.3)
    for j in range(len(rider_metrics), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'rider_metrics_distribution.png'))
    plt.close()

    print(f"Visualizations saved to: {ANALYSIS_DIR}")


def compare_with_zomato(simulation_file=None):
    print("\n" + "=" * 60)
    print("Comparing Simulation with Real Experimental Data")
    print("=" * 60)

    if simulation_file is None:
        simulation_file = os.path.join(SIMULATION_DIR, 'aggregated_results.csv')

    if not os.path.exists(simulation_file):
        print(f"Simulation results not found: {simulation_file}")
        return

    sim_df = pd.read_csv(simulation_file)

    exp_dirs = sorted([d for d in glob.glob(os.path.join(BASE_DIR, 'exp*'))
                      if os.path.isdir(d) and d.split('exp')[-1].isdigit()])

    if not exp_dirs:
        print("No experimental data found in exp*/ directories")
        return

    print(f"Found {len(exp_dirs)} experimental datasets: {[os.path.basename(d) for d in exp_dirs]}")

    exp_metrics = _load_experimental_metrics(exp_dirs)

    sim_rider_mean = {
        'money': sim_df['money'].mean(),
        'labor': sim_df['labor'].mean(),
        'total_order': sim_df['total_order'].mean(),
        'utility': sim_df['utility'].mean()
    }

    print("\nComparison (Simulation vs Experimental):")
    print(f"{'Metric':<20} | {'Simulation':<15} | {'Experimental':<15} | {'Diff %':<10}")
    print("-" * 65)

    for metric in ['money', 'labor', 'total_order', 'utility']:
        sim_val = sim_rider_mean.get(metric, 0)
        exp_val = exp_metrics.get(metric, 0)
        diff_pct = ((sim_val - exp_val) / exp_val * 100) if exp_val != 0 else 0
        print(f"{metric:<20} | {sim_val:<15.2f} | {exp_val:<15.2f} | {diff_pct:>+.2f}%")


def _load_experimental_metrics(exp_dirs):
    all_rider_data = []

    for exp_dir in exp_dirs:
        rider_files = glob.glob(os.path.join(exp_dir, 'deliver_*.csv'))
        for rider_file in rider_files:
            try:
                rider_df = pd.read_csv(rider_file)
                if 'money' in rider_df.columns and 'dis' in rider_df.columns:
                    all_rider_data.append({
                        'money': rider_df['money'].iloc[-1] if len(rider_df) > 0 else 0,
                        'labor': rider_df['dis'].iloc[-1] if len(rider_df) > 0 else 0,
                    })
            except Exception:
                continue

    if not all_rider_data:
        return {}

    metrics_df = pd.DataFrame(all_rider_data)
    return {
        'money': metrics_df['money'].mean(),
        'labor': metrics_df['labor'].mean() if 'labor' in metrics_df else metrics_df['dis'].mean(),
        'total_order': 0,
        'utility': 0
    }


def compare_with_real_data(simulation_file=None):
    """Backward-compatible alias for the paper comparison function."""

    return compare_with_zomato(simulation_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Simulation Results')
    parser.add_argument('--compare', action='store_true', help='Compare with real experimental data')
    parser.add_argument('--simulation_file', type=str, default=None, help='Path to simulation results')

    args = parser.parse_args()

    analyze_simulation_results()

    if args.compare:
        compare_with_zomato(args.simulation_file)


if __name__ == '__main__':
    main()
