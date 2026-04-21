import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import math


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'analysis_results')


def calculate_distance(x1, y1, x2, y2):
    try:
        return math.sqrt((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2)
    except (ValueError, TypeError):
        return 0.0


def analyze_multi_source_data():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sim_dir = os.path.join(BASE_DIR, 'simulation_results')
    sim_agg_file = os.path.join(sim_dir, 'aggregated_results.csv')

    exp_dirs = sorted([d for d in glob.glob(os.path.join(BASE_DIR, 'exp*'))
                      if os.path.isdir(d) and d.split('exp')[-1].isdigit()])

    all_data_sources = []

    if os.path.exists(sim_agg_file):
        print("Loading simulation data...")
        sim_df = pd.read_csv(sim_agg_file)
        sim_df['source'] = 'simulation'
        sim_df['run_id'] = 'sim_' + sim_df['run_id'].astype(str)
        all_data_sources.append(('simulation', sim_df))

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"Loading experimental data: {exp_name}")
        exp_data = load_exp_data(exp_dir)
        if exp_data is not None:
            exp_data['source'] = exp_name
            all_data_sources.append((exp_name, exp_data))

    if len(all_data_sources) < 2:
        print("Need at least 2 data sources to compare")
        return

    print(f"\nLoaded {len(all_data_sources)} data sources")

    for name, df in all_data_sources:
        print(f"\n{name}: {len(df)} rider records")
        print(f"  Columns: {list(df.columns)}")

    _compare_sources(all_data_sources)
    _generate_comparison_plots(all_data_sources)


def load_exp_data(exp_dir):
    rider_files = glob.glob(os.path.join(exp_dir, 'deliver_*.csv'))

    if not rider_files:
        return None

    all_rider_data = []
    revenue_factor = 10
    cost_factor = 0.08

    for rider_file in rider_files:
        try:
            df = pd.read_csv(rider_file)
            if not all(col in df.columns for col in ['x', 'y', 'money', 'dis']):
                continue
            if df.empty or len(df) < 2:
                continue

            df['x'] = pd.to_numeric(df['x'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df['money'] = pd.to_numeric(df['money'], errors='coerce')
            df['dis'] = pd.to_numeric(df['dis'], errors='coerce')
            df.dropna(subset=['x', 'y', 'money', 'dis'], inplace=True)

            if len(df) < 2:
                continue

            basename = os.path.basename(rider_file)
            parts = basename.split('.')[0].split('_')
            rider_id = int(parts[1])

            total_distance = df['dis'].iloc[-1] if 'dis' in df.columns else 0
            last_revenue = df['money'].iloc[-1] if 'money' in df.columns else 0

            reward_scaled = revenue_factor * last_revenue
            cost_scaled = cost_factor * total_distance
            individual_utility = reward_scaled - cost_scaled

            efficiency = last_revenue / total_distance if total_distance > 1e-9 else 0

            all_rider_data.append({
                'rider_id': rider_id,
                'money': last_revenue,
                'labor': total_distance,
                'total_order': df['order_count'].iloc[-1] if 'order_count' in df.columns else 0,
                'efficiency': efficiency,
                'utility': individual_utility
            })
        except Exception:
            continue

    if not all_rider_data:
        return None

    df_result = pd.DataFrame(all_rider_data)
    return df_result


def _compare_sources(all_data_sources):
    print("\n" + "=" * 70)
    print("Cross-Source Comparison")
    print("=" * 70)

    metrics = ['money', 'labor', 'total_order', 'efficiency', 'utility']

    comparison_data = []
    for source_name, df in all_data_sources:
        if 'utility' not in df.columns and 'efficiency' in df.columns:
            df['utility'] = df['efficiency']
        row = {'source': source_name}
        for metric in metrics:
            if metric in df.columns:
                row[f'{metric}_mean'] = df[metric].mean()
                row[f'{metric}_std'] = df[metric].std()
            else:
                row[f'{metric}_mean'] = 0
                row[f'{metric}_std'] = 0
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    print("\nMetric Summary (Mean ± Std):")
    print("-" * 70)
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        for _, row in comparison_df.iterrows():
            mean_val = row.get(f'{metric}_mean', 0)
            std_val = row.get(f'{metric}_std', 0)
            print(f"  {row['source']:<20}: {mean_val:>10.4f} ± {std_val:<10.4f}")

    comparison_file = os.path.join(RESULTS_DIR, 'multi_source_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison saved to: {comparison_file}")


def _generate_comparison_plots(all_data_sources):
    print("\nGenerating comparison plots...")

    metrics = ['money', 'labor', 'total_order', 'efficiency', 'utility']
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_data_sources)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        plot_data = []
        labels = []

        for (source_name, df), color in zip(all_data_sources, colors):
            if metric in df.columns:
                values = df[metric].dropna().values
                if len(values) > 0:
                    plot_data.append(values)
                    labels.append(source_name)

        if plot_data:
            bp = ax.boxplot(plot_data, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(plot_data)]):
                patch.set_facecolor(color)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Distribution')
            ax.grid(axis='y', alpha=0.3)

    axes[-1].axis('off')

    plt.tight_layout()
    plot_file = os.path.join(RESULTS_DIR, 'multi_source_comparison.png')
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {plot_file}")


def main():
    analyze_multi_source_data()


if __name__ == '__main__':
    main()
