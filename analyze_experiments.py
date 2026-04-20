import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import math

# 主数据目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 结果保存目录
RESULTS_DIR = os.path.join(BASE_DIR, 'analysis_results')

# 创建结果目录（如果不存在）
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def calculate_distance(x1, y1, x2, y2):
    """计算两点之间的欧氏距离"""
    # Handle potential non-numeric inputs gracefully
    try:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    except (ValueError, TypeError):
        # print(f"Warning: Invalid coordinates found ({x1}, {y1}), ({x2}, {y2}). Returning distance 0.")
        return 0.0


def analyze_experiment(exp_dir):
    """分析单个实验文件夹的数据 - Corrected based on the paper"""
    print(f"Analyzing experiment: {os.path.basename(exp_dir)}")
    rider_files = glob.glob(os.path.join(exp_dir, 'deliver_*.csv'))

    # 定义收入和成本的系数 (These act as scaling factors for Reward and Cost)
    revenue_factor = 10
    cost_factor = 0.08

    all_rider_data = []
    total_system_revenue_unscaled = 0 # Sum of raw 'money'
    total_system_cost_unscaled = 0    # Sum of raw 'distance'

    if not rider_files:
        print(f"  No rider data found in {os.path.basename(exp_dir)}.")
        return None, None # 返回空值表示没有数据

    for rider_file in rider_files:
        try:
            df = pd.read_csv(rider_file)
            # Ensure 'x', 'y', 'money' columns exist
            if not all(col in df.columns for col in ['x', 'y', 'money']):
                 print(f"  Skipping file due to missing columns: {os.path.basename(rider_file)}")
                 continue
            if df.empty or len(df) < 2:
                print(f"  Skipping empty or single-row file: {os.path.basename(rider_file)}")
                continue

            # Drop rows with NaN in essential columns before calculation
            df.dropna(subset=['x', 'y', 'money'], inplace=True)
            if len(df) < 2:
                 print(f"  Skipping file after dropping NaN (too few rows): {os.path.basename(rider_file)}")
                 continue

             # Ensure correct data types
            df['x'] = pd.to_numeric(df['x'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df['money'] = pd.to_numeric(df['money'], errors='coerce')
            df.dropna(subset=['x', 'y', 'money'], inplace=True) # Drop again if coercion failed
            if len(df) < 2:
                 print(f"  Skipping file after numeric coercion (too few rows): {os.path.basename(rider_file)}")
                 continue

            # 提取骑手ID
            try:
                # Assuming filename like 'deliver_ID_...' or 'deliver_ID.csv'
                basename = os.path.basename(rider_file)
                parts = basename.split('.')[0].split('_')
                rider_id = int(parts[1]) # Expecting ID as the second part
            except (IndexError, ValueError):
                rider_id = os.path.basename(rider_file) # Fallback to filename

            # --- 计算个体指标 ---
            df = df.reset_index(drop=True) # Ensure default integer index

            # -- 分析骑手行为 (移动, 挣钱, 休息) --
            move_steps = 0
            earn_steps = 0
            rest_steps = 0
            total_distance = 0.0
            action_threshold = 0.1 # 移动距离阈值

            if len(df) > 1:
                # Calculate differences between consecutive rows
                df['prev_x'] = df['x'].shift(1)
                df['prev_y'] = df['y'].shift(1)
                df['prev_money'] = df['money'].shift(1)

                # Calculate distance and money change for each step (starting from the second step)
                # Use apply with a lambda function to handle potential errors row-wise
                df['delta_dist'] = df.apply(lambda row: calculate_distance(row['prev_x'], row['prev_y'], row['x'], row['y']) if pd.notna(row['prev_x']) else 0.0, axis=1)
                df['delta_money'] = df.apply(lambda row: row['money'] - row['prev_money'] if pd.notna(row['prev_money']) else 0.0, axis=1)

                # Classify actions for steps 1 to N-1
                for i in range(1, len(df)):
                    delta_dist = df.loc[i, 'delta_dist']
                    delta_money = df.loc[i, 'delta_money']
                    total_distance += delta_dist # Accumulate total distance here

                    if delta_dist > action_threshold:
                        move_steps += 1
                    elif delta_money > 1e-9: # Check for positive money change
                        earn_steps += 1
                    else:
                        rest_steps += 1
            # Handle single-step case (or if loop doesn't run)
            if len(df) == 1:
                 rest_steps = 1 # Assume resting if only one data point

            # -- 计算其他指标 --
            # 计算总收益 (Reward_i - using last step's cumulative money)
            last_revenue = df['money'].iloc[-1] if pd.api.types.is_numeric_dtype(df['money']) and not df['money'].empty else 0

            # 计算骑手个体效用 (IndividualUtility_i = Reward - Cost)
            reward_scaled = revenue_factor * last_revenue
            cost_scaled = cost_factor * total_distance
            individual_utility = reward_scaled - cost_scaled

            # 计算骑手效能 (Old metric, not from paper's core model)
            efficiency = last_revenue / total_distance if total_distance > 1e-9 else 0

            all_rider_data.append({
                'rider_id': rider_id,
                'cost': total_distance,          # Unscaled cost (calculated during action analysis)
                'revenue': last_revenue,         # Unscaled revenue
                'efficiency': efficiency,        # Old efficiency metric
                'individual_utility': individual_utility, # Paper Eq (1) based metric
                'move_steps': move_steps,
                'earn_steps': earn_steps,
                'rest_steps': rest_steps,
                'total_steps': len(df) # Total steps recorded for the rider
            })

            total_system_revenue_unscaled += last_revenue
            total_system_cost_unscaled += total_distance

        except pd.errors.EmptyDataError:
             print(f"  Skipping empty file: {os.path.basename(rider_file)}")
        except Exception as e:
            print(f"  Error processing file {os.path.basename(rider_file)}: {e}")
            # Optionally, print traceback for detailed debugging
            # import traceback
            # traceback.print_exc()

    if not all_rider_data:
        print(f"  No valid rider data processed in {os.path.basename(exp_dir)}.")
        return None, None

    rider_df = pd.DataFrame(all_rider_data)
    exp_name = os.path.basename(exp_dir)

    # --- 计算系统级指标 (Based on Paper) --- #
    N_total = len(rider_df) # Total number of agents/riders processed

    # 1. 生产力 (Productivity): Paper Eq (16)
    # Sum of individual utilities
    productivity = rider_df['individual_utility'].sum()

    # 2. 价值熵 (Value Entropy): Paper Eq (13)
    value_entropy = 0.0 # Default value
    H_T = 0.0         # Current Entropy, Paper Eq (7)
    H_best = 0.0      # Optimal Entropy, Paper Eq (12)

    if N_total > 0:
        # Calculate H_best: Paper Eq (12)
        if N_total == 1:
             H_best = 0.0 # log2(sqrt(1)) = 0
        else:
             # Use np.log2 for base 2 logarithm
             H_best = np.log2(np.sqrt(N_total))

        # Calculate H_T: Paper Eq (7) based on niches/bins
        if N_total > 1 : # Entropy requires probability distribution, meaningful for N > 1
            # Determine number of niches/bins (m). Using sqrt(N) as a heuristic based on H_best's structure.
            # Ceiling ensures at least 1 bin. Adjust if needed.
            num_bins = max(1, int(np.ceil(np.sqrt(N_total))))

            utilities = rider_df['individual_utility']

            # Handle case where all utilities are the same (results in 1 bin)
            if utilities.nunique() == 1:
                 num_bins = 1
                 counts = [N_total]
            else:
                 # Use numpy histogram to get counts (Nj) per bin (niche)
                 # bins+1 edges define bins number of bins
                 try:
                     counts, bin_edges = np.histogram(utilities, bins=num_bins)
                 except Exception as hist_err:
                      print(f"  Warning: Histogram calculation failed for {exp_name}: {hist_err}. Setting H_T to 0.")
                      counts = [] # Ensure H_T becomes 0


            # Calculate probabilities pj = Nj / N_total
            probabilities = counts / N_total # Broadcasting N_total

            # Calculate H_T = - sum(pj * log2(pj)) for pj > 0
            # Filter out zero probabilities to avoid log2(0)
            valid_probabilities = probabilities[probabilities > 1e-9]
            if len(valid_probabilities) > 0:
                H_T = -np.sum(valid_probabilities * np.log2(valid_probabilities))
            else:
                H_T = 0.0 # If no valid probabilities (e.g., N_total=0, or counts issue)

        elif N_total == 1:
            # Single agent: One niche, p1 = 1/1 = 1. H_T = -1*log2(1) = 0.
             H_T = 0.0


        # Calculate Value Entropy: Paper Eq (13)
        # Handle H_best = 0 case (happens when N_total = 1)
        if H_best < 1e-9: # If H_best is effectively zero
             if abs(H_T - H_best) < 1e-9: # If H_T is also zero (perfect match)
                 value_entropy = np.exp(1.0) # Max value entropy as per paper Table I logic (Scenario B)
             else:
                 value_entropy = np.exp(-np.inf) # Or 0. Very low if H_T differs from H_best=0
                 # Or set to 0 as a floor value? Let's use 0 for practical plotting.
                 value_entropy = 0.0
        else: # Normal case: H_best > 0
             relative_diff = np.abs(H_T - H_best) / H_best
             value_entropy = np.exp(1.0 - relative_diff)

    else: # N_total = 0
        productivity = 0.0
        value_entropy = 0.0
        H_T = 0.0
        H_best = 0.0


    # 3. 系统效用 (System Utility): Paper Eq (15)
    system_utility = value_entropy * productivity

    # --- Old System Efficiency (For comparison) ---
    # Based on unscaled total revenue and cost
    system_efficiency_old = total_system_revenue_unscaled / total_system_cost_unscaled if total_system_cost_unscaled > 1e-9 else 0

    # --- Results Saving and Visualization --- #

    # Create subdirectory for this experiment's plots if it doesn't exist
    exp_plot_dir = os.path.join(RESULTS_DIR, exp_name)
    if not os.path.exists(exp_plot_dir):
        os.makedirs(exp_plot_dir)

    # 1. Plot: Rider Individual Utility Distribution (More relevant than old efficiency)
    plt.figure(figsize=(12, 6))
    plt.hist(rider_df['individual_utility'], bins=15, edgecolor='black') # More bins might be useful
    plt.xlabel('Individual Utility (Scaled Reward - Scaled Cost)')
    plt.ylabel('Number of Riders')
    plt.title(f'Distribution of Individual Utility in {exp_name}\n(Productivity: {productivity:.2f})')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plot_path_rider_util = os.path.join(exp_plot_dir, f'{exp_name}_individual_utility_distribution.png')
    try:
        plt.savefig(plot_path_rider_util)
    except Exception as save_err:
        print(f"  Error saving plot {plot_path_rider_util}: {save_err}")
    plt.close()
    # print(f"  Saved individual utility distribution plot to: {plot_path_rider_util}")

    # (Optional) Plot: Old Rider Efficiency Bar Chart (Keep if useful)
    plt.figure(figsize=(max(15, N_total * 0.3), 7)) # Dynamic width
    try:
        rider_df_sorted = rider_df.sort_values('rider_id')
        # Convert rider IDs to string for categorical plotting
        rider_ids_str = rider_df_sorted['rider_id'].astype(str)
        plt.bar(rider_ids_str, rider_df_sorted['efficiency'])
        plt.xlabel('Rider ID')
        plt.ylabel('Efficiency (Unscaled Revenue / Unscaled Distance)')
        plt.title(f'Rider Efficiency (Old Metric) in {exp_name}\nSystem Efficiency (Old): {system_efficiency_old:.4f}')
        # Rotate ticks only if many riders
        if N_total > 20:
            plt.xticks(rotation=90, fontsize=8)
        else:
             plt.xticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plot_path_rider_eff = os.path.join(exp_plot_dir, f'{exp_name}_rider_efficiency_old.png')
        plt.savefig(plot_path_rider_eff)
    except Exception as plot_err:
        print(f"  Warning: Could not plot old rider efficiency for {exp_name}: {plot_err}")
    finally:
        plt.close() # Ensure plot is closed even if saving fails
    # print(f"  Saved old rider efficiency plot to: {plot_path_rider_eff}")


    # --- Return Experiment Summary ---
    experiment_summary = {
        'experiment': exp_name,
        'total_riders': N_total,
        # Using unscaled totals for reporting basic revenue/cost
        'total_revenue_unscaled': total_system_revenue_unscaled,
        'total_cost_unscaled': total_system_cost_unscaled,
        'system_efficiency_old': system_efficiency_old, # Old metric
        # Metrics based on the paper
        'productivity': productivity,
        'H_T': H_T, # Current Entropy
        'H_best': H_best, # Optimal Entropy
        'value_entropy': value_entropy,
        'system_utility': system_utility
    }
    return experiment_summary, rider_df

# --- 主程序 --- #
all_experiments_summary = []

# 遍历所有 exp* 文件夹
base_dir_content = os.listdir(BASE_DIR)
exp_dirs = sorted([d for d in base_dir_content if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith('exp')])

# Limit number of experiments for debugging if necessary
# exp_dirs = exp_dirs[:5]

for item in exp_dirs:
    item_path = os.path.join(BASE_DIR, item)
    print("-" * 40) # Separator for experiments
    summary, rider_data_df = analyze_experiment(item_path) # Get both summary and detailed data
    if summary:
        all_experiments_summary.append(summary)
        # Optional: Save detailed rider data for each experiment
        if rider_data_df is not None and not rider_data_df.empty:
             rider_csv_path = os.path.join(RESULTS_DIR, f'{summary["experiment"]}_rider_details.csv')
             try:
                  rider_data_df.to_csv(rider_csv_path, index=False)
                  # print(f"  Saved rider details to: {rider_csv_path}")
             except Exception as csv_err:
                  print(f"  Error saving rider details CSV for {summary['experiment']}: {csv_err}")


# --- 跨实验分析与绘图 --- #
if all_experiments_summary:
    summary_df = pd.DataFrame(all_experiments_summary)
    # Ensure experiment column is suitable for sorting if needed (e.g., numerical part)
    try:
        # Attempt to extract number from experiment name for sorting
        summary_df['exp_num'] = summary_df['experiment'].str.extract(r'exp(\d+)').astype(int)
        summary_df = summary_df.sort_values('exp_num').drop(columns=['exp_num'])
    except:
        # Fallback to alphabetical sort if extraction fails
        summary_df = summary_df.sort_values('experiment')

    print("\n--- Experiment Summary ---")
    print(summary_df)
    print("-" * 60)

    # --- Plotting Comparisons ---

    plot_metrics = [
        ('productivity', 'Productivity (Sum of Individual Utilities)', 'productivity_comparison.png'),
        ('value_entropy', 'Value Entropy (Paper Eq 13)', 'value_entropy_comparison.png'),
        ('system_utility', 'System Utility (Paper Eq 15)', 'system_utility_comparison.png'),
        ('system_efficiency_old', 'System Efficiency (Old: Total Revenue / Total Cost)', 'system_efficiency_old_comparison.png'),
        ('H_T', 'Current Entropy (H_T)', 'HT_comparison.png'),
        ('H_best', 'Optimal Entropy (H_best)', 'Hbest_comparison.png'),
    ]

    for metric, title, filename in plot_metrics:
        if metric not in summary_df.columns:
            print(f"Warning: Metric '{metric}' not found in summary DataFrame. Skipping plot.")
            continue

        plt.figure(figsize=(max(10, len(summary_df) * 0.5), 6)) # Dynamic width
        plt.bar(summary_df['experiment'], summary_df[metric])
        plt.xlabel('Experiment')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{title} Comparison')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        # Set y-axis floor to 0 for non-negative metrics if appropriate
        if metric in ['value_entropy', 'system_efficiency_old', 'H_T', 'H_best']:
             plt.ylim(bottom=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, filename)
        try:
            plt.savefig(plot_path)
            print(f"Saved plot: {plot_path}")
        except Exception as save_err:
            print(f"  Error saving plot {plot_path}: {save_err}")
        plt.close()


    # --- Save Overall Summary to CSV ---
    summary_csv_path = os.path.join(RESULTS_DIR, 'experiments_summary_corrected.csv')
    try:
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved experiments summary to: {summary_csv_path}")
    except Exception as csv_err:
        print(f"  Error saving summary CSV: {csv_err}")

else:
    print("No experiment data was successfully analyzed.")

print("\nAnalysis complete.")
