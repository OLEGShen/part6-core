#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import math
import re
from collections import Counter
import ast # For safely evaluating string representations of lists
import numpy as np # Ensure numpy is imported

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'analysis_results')
llm_analysis_candidates = [
    os.path.join(RESULTS_DIR, 'llm_thought_analysis.csv'),
    os.path.join(RESULTS_DIR, 'llm_thought_analysis_exp1_realtime.csv')
]
LLM_ANALYSIS_FILE = next((path for path in llm_analysis_candidates if os.path.exists(path)), llm_analysis_candidates[0])
EXPERIMENT_DIRS_PATTERN = os.path.join(BASE_DIR, 'exp*')
TOP_N_INTENTIONS = 5
ACTION_DIST_THRESHOLD = 0.1 # Distance threshold to consider 'move'
ACTION_MONEY_THRESHOLD = 1e-9 # Money threshold to consider 'earn'

# Ensure output directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Helper Functions ---
def calculate_distance(x1, y1, x2, y2):
    """Calculates Euclidean distance between two points."""
    try:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    except (ValueError, TypeError):
        return 0.0

def safe_literal_eval(val):
    """Safely evaluate a string literal (like a list) or return empty list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            # Basic check for list-like string
            if val.startswith('[') and val.endswith(']'):
                 # Replace single quotes with double quotes for valid JSON-like format if needed
                 # Be cautious with complex strings, might need more robust parsing
                 # val_corrected = val.replace("'", '"')
                 # For now, rely on ast.literal_eval which is safer for Python literals
                return ast.literal_eval(val)
            else:
                # Handle simple comma-separated strings if not list format
                return [item.strip() for item in val.split(',') if item.strip()]
        except (ValueError, SyntaxError, TypeError):
            # If eval fails, return empty list or handle as single keyword
            # print(f"Warning: Could not parse keywords: {val}")
            if val: # Treat as single keyword if not empty and not parsable as list
                return [val.strip()]
            return []
    return [] # Return empty list for other types or None

# --- Main Analysis Logic ---

# 1. Load LLM Thought Analysis Data
print(f"Loading LLM analysis data from: {LLM_ANALYSIS_FILE}")
try:
    llm_df = pd.read_csv(LLM_ANALYSIS_FILE)
    # Ensure required columns exist
    required_llm_cols = ['Experiment', 'RiderID', 'Step', 'RationalKeywords', 'EmotionalKeywords']
    if not all(col in llm_df.columns for col in required_llm_cols):
        print(f"Error: LLM analysis file missing required columns: {required_llm_cols}")
        exit(1)
    print(f"Loaded {len(llm_df)} rows from LLM analysis.")
    # Convert keyword strings to lists
    llm_df['RationalKeywords'] = llm_df['RationalKeywords'].apply(safe_literal_eval)
    llm_df['EmotionalKeywords'] = llm_df['EmotionalKeywords'].apply(safe_literal_eval)

except FileNotFoundError:
    print(f"Error: LLM analysis file not found at {LLM_ANALYSIS_FILE}")
    print("Please ensure '1_analyze_thoughts_with_llm.py' has been run successfully.")
    exit(1)
except Exception as e:
    print(f"Error loading LLM analysis file: {e}")
    exit(1)

if llm_df.empty:
    print("Error: LLM analysis data is empty. Cannot proceed.")
    exit(1)

# 2. Load and Process Rider Action Data Step-by-Step
print("Loading and processing rider action data...")
all_step_data = []
experiment_dirs = glob.glob(EXPERIMENT_DIRS_PATTERN)

for exp_dir in experiment_dirs:
    if not os.path.isdir(exp_dir):
        continue
    exp_name = os.path.basename(exp_dir)
    print(f"  Processing experiment: {exp_name}")
    rider_files = glob.glob(os.path.join(exp_dir, 'deliver_*.csv'))

    for rider_file in rider_files:
        try:
            rider_id_match = re.search(r'deliver_(\d+)_record.csv', os.path.basename(rider_file))
            if not rider_id_match:
                continue
            rider_id = int(rider_id_match.group(1))

            df = pd.read_csv(rider_file)
            if df.empty or len(df) < 2:
                continue

            # Ensure necessary columns and types
            df['x'] = pd.to_numeric(df['x'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df['money'] = pd.to_numeric(df['money'], errors='coerce')
            df.dropna(subset=['x', 'y', 'money'], inplace=True)
            if len(df) < 2:
                continue

            df = df.reset_index().rename(columns={'index': 'Step'}) # Use original index as Step

            # Calculate deltas
            df['prev_x'] = df['x'].shift(1)
            df['prev_y'] = df['y'].shift(1)
            df['prev_money'] = df['money'].shift(1)

            df['delta_dist'] = df.apply(lambda row: calculate_distance(row['prev_x'], row['prev_y'], row['x'], row['y']) if pd.notna(row['prev_x']) else 0.0, axis=1)
            df['delta_money'] = df.apply(lambda row: row['money'] - row['prev_money'] if pd.notna(row['prev_money']) else 0.0, axis=1)

            # Determine actions for each step (starting from step 1)
            df['Action_Move'] = df['delta_dist'] > ACTION_DIST_THRESHOLD
            df['Action_Earn'] = df['delta_money'] > ACTION_MONEY_THRESHOLD

            # Select relevant columns and add identifiers
            step_df = df[['Step', 'Action_Move', 'Action_Earn']].copy()
            step_df['Experiment'] = exp_name
            step_df['RiderID'] = rider_id

            # Handle step 0: Assume not moving and not earning initially
            step_0_data = pd.DataFrame([{'Step': 0, 'Action_Move': False, 'Action_Earn': False, 'Experiment': exp_name, 'RiderID': rider_id}])
            # Correct the actions for step > 0 based on delta from previous
            step_df = step_df[step_df['Step'] > 0] # Actions are defined by change from previous step

            # Combine step 0 and subsequent steps
            # Note: The LLM data might start from step 0 or 1. Need to align.
            # Let's assume LLM data aligns with the step index from the CSV.
            # We need actions *at* step t, and thoughts *at* step t.
            # Let's redefine actions based on the state *at* step t, derived from change *leading to* step t.
            # Re-calculate actions based on the change leading TO that step
            action_data_for_merge = []
            # Step 0: Assume initial state
            action_data_for_merge.append({'Step': 0, 'Action_Move': False, 'Action_Earn': False, 'Experiment': exp_name, 'RiderID': rider_id})
            # Steps 1 onwards:
            for i in range(1, len(df)):
                 action_data_for_merge.append({
                     'Step': df.loc[i, 'Step'],
                     'Action_Move': df.loc[i, 'delta_dist'] > ACTION_DIST_THRESHOLD,
                     'Action_Earn': df.loc[i, 'delta_money'] > ACTION_MONEY_THRESHOLD,
                     'Experiment': exp_name,
                     'RiderID': rider_id
                 })

            all_step_data.extend(action_data_for_merge)

        except Exception as e:
            print(f"    Error processing file {os.path.basename(rider_file)}: {e}")

if not all_step_data:
    print("Error: No rider action data could be processed. Cannot proceed.")
    exit(1)

actions_df = pd.DataFrame(all_step_data)
print(f"Processed action data for {actions_df['Experiment'].nunique()} experiments, {actions_df['RiderID'].nunique()} unique riders.")

# 3. Merge Action Data with LLM Thought Data
print("Merging action and thought data...")
# Ensure correct types for merging
llm_df['RiderID'] = llm_df['RiderID'].astype(int)
llm_df['Step'] = llm_df['Step'].astype(int)
actions_df['RiderID'] = actions_df['RiderID'].astype(int)
actions_df['Step'] = actions_df['Step'].astype(int)

merged_df = pd.merge(actions_df, llm_df, on=['Experiment', 'RiderID', 'Step'], how='inner')

if merged_df.empty:
    print("Error: Merging actions and thoughts resulted in an empty DataFrame. Check alignment of Experiment, RiderID, Step.")
    print(f"Action Steps Info: {actions_df[['Experiment', 'RiderID', 'Step']].head()}")
    print(f"LLM Steps Info: {llm_df[['Experiment', 'RiderID', 'Step']].head()}")
    exit(1)

print(f"Merged data contains {len(merged_df)} steps with both action and thought info.")

# 4. Identify Top Intentions
print("Identifying top intentions...")
all_rational_keywords = [kw for sublist in merged_df['RationalKeywords'] for kw in sublist]
all_emotional_keywords = [kw for sublist in merged_df['EmotionalKeywords'] for kw in sublist]

rational_counts = Counter(all_rational_keywords)
emotional_counts = Counter(all_emotional_keywords)

top_rational = [kw for kw, count in rational_counts.most_common(TOP_N_INTENTIONS)]
top_emotional = [kw for kw, count in emotional_counts.most_common(TOP_N_INTENTIONS)]

print(f"  Top {TOP_N_INTENTIONS} Rational Keywords: {top_rational}")
print(f"  Top {TOP_N_INTENTIONS} Emotional Keywords: {top_emotional}")

if not top_rational and not top_emotional:
    print("Warning: Could not identify top keywords. Analysis might be limited.")

# 5. Analyze Intention-Behavior Co-occurrence (State-based)
print("Analyzing intention-behavior co-occurrence (state-based)...")
state_correlation_data = {'rational': {}, 'emotional': {}}
action_state_cols = ['Action_Move_True', 'Action_Move_False', 'Action_Earn_True', 'Action_Earn_False']
prob_state_cols = ['Prob_Move_True', 'Prob_Move_False', 'Prob_Earn_True', 'Prob_Earn_False']

for kw_type, top_list in [('rational', top_rational), ('emotional', top_emotional)]:
    for kw in top_list:
        # Filter steps where the keyword is present
        # Ensure keywords are treated as sets for robust checking
        kw_present_mask = merged_df[f'{kw_type.capitalize()}Keywords'].apply(lambda kws: kw in set(kws))
        kw_present_df = merged_df[kw_present_mask]
        total_kw_steps = len(kw_present_df)
        state_correlation_data[kw_type][kw] = {'Total_Keyword_Steps': total_kw_steps}

        if total_kw_steps > 0:
            # Count co-occurrences with action states
            state_correlation_data[kw_type][kw]['Action_Move_True'] = len(kw_present_df[kw_present_df['Action_Move'] == True])
            state_correlation_data[kw_type][kw]['Action_Move_False'] = len(kw_present_df[kw_present_df['Action_Move'] == False])
            state_correlation_data[kw_type][kw]['Action_Earn_True'] = len(kw_present_df[kw_present_df['Action_Earn'] == True])
            state_correlation_data[kw_type][kw]['Action_Earn_False'] = len(kw_present_df[kw_present_df['Action_Earn'] == False])

            # Calculate probabilities P(Action State | Intention Presence)
            state_correlation_data[kw_type][kw]['Prob_Move_True'] = state_correlation_data[kw_type][kw]['Action_Move_True'] / total_kw_steps
            state_correlation_data[kw_type][kw]['Prob_Move_False'] = state_correlation_data[kw_type][kw]['Action_Move_False'] / total_kw_steps
            state_correlation_data[kw_type][kw]['Prob_Earn_True'] = state_correlation_data[kw_type][kw]['Action_Earn_True'] / total_kw_steps
            state_correlation_data[kw_type][kw]['Prob_Earn_False'] = state_correlation_data[kw_type][kw]['Action_Earn_False'] / total_kw_steps
        else:
             for prob_col in prob_state_cols:
                 state_correlation_data[kw_type][kw][prob_col] = 0.0
             for count_col in action_state_cols:
                  state_correlation_data[kw_type][kw][count_col] = 0

# Convert state correlation data to DataFrame for easier access
state_correlation_scores = {'rational': pd.DataFrame(), 'emotional': pd.DataFrame()}
for kw_type in ['rational', 'emotional']:
     scores_list = []
     for kw, data in state_correlation_data[kw_type].items():
         row_data = {'Intention': kw}
         row_data.update(data)
         scores_list.append(row_data)
     if scores_list:
         state_correlation_scores[kw_type] = pd.DataFrame(scores_list).set_index('Intention')

print("\nRational Intention - Action State Correlation (Probability P(Action State | Intention Presence)):")
if not state_correlation_scores['rational'].empty:
    print(state_correlation_scores['rational'][prob_state_cols])
else:
    print("No rational state correlation data.")

print("\nEmotional Intention - Action State Correlation (Probability P(Action State | Intention Presence)):")
if not state_correlation_scores['emotional'].empty:
    print(state_correlation_scores['emotional'][prob_state_cols])
else:
    print("No emotional state correlation data.")


# --- Keep the original change-based analysis for reference or potential future use --- 
# 5. Analyze Intention-Behavior Changes (Original - Co-occurrence of *Changes*)
print("\nAnalyzing intention-behavior co-occurrence (change-based)...")
correlation_data = {'rational': {}, 'emotional': {}}

# Initialize correlation counters
for kw_type, top_list in [('rational', top_rational), ('emotional', top_emotional)]:
    for kw in top_list:
        correlation_data[kw_type][kw] = {'Move_Change': 0, 'Earn_Change': 0, 'Total_Intention_Changes': 0}

# Sort data for sequential analysis per rider
merged_df = merged_df.sort_values(by=['Experiment', 'RiderID', 'Step'])

# Iterate through steps for each rider
for _, rider_group in merged_df.groupby(['Experiment', 'RiderID']):
    rider_group = rider_group.reset_index()
    for i in range(len(rider_group) - 1):
        step_t = rider_group.loc[i]
        step_t1 = rider_group.loc[i+1]

        # Check for action changes
        move_changed = step_t['Action_Move'] != step_t1['Action_Move']
        earn_changed = step_t['Action_Earn'] != step_t1['Action_Earn']

        # Check for intention changes (specific top keywords appearing/disappearing)
        keywords_t = {'rational': set(step_t['RationalKeywords']), 'emotional': set(step_t['EmotionalKeywords'])}
        keywords_t1 = {'rational': set(step_t1['RationalKeywords']), 'emotional': set(step_t1['EmotionalKeywords'])}

        for kw_type, top_list in [('rational', top_rational), ('emotional', top_emotional)]:
            for kw in top_list:
                intention_changed = (kw in keywords_t and kw not in keywords_t1) or \
                                  (kw not in keywords_t and kw in keywords_t1)

                if intention_changed:
                    correlation_data[kw_type][kw]['Total_Intention_Changes'] += 1
                    if move_changed:
                        correlation_data[kw_type][kw]['Move_Change'] += 1
                    if earn_changed:
                        correlation_data[kw_type][kw]['Earn_Change'] += 1

# 6. Calculate Correlation Scores (Co-occurrence Probability)
print("Calculating correlation scores...")
correlation_scores = {'rational': pd.DataFrame(), 'emotional': pd.DataFrame()}

for kw_type in ['rational', 'emotional']:
    scores_list = []
    for kw, data in correlation_data[kw_type].items():
        total_changes = data['Total_Intention_Changes']
        if total_changes > 0:
            score_move = data['Move_Change'] / total_changes
            score_earn = data['Earn_Change'] / total_changes
        else:
            score_move = 0
            score_earn = 0
        scores_list.append({
            'Intention': kw,
            'Prob_Move_Change': score_move,
            'Prob_Earn_Change': score_earn,
            'Total_Intention_Changes': total_changes,
            'Move_Changes_Count': data['Move_Change'],
            'Earn_Changes_Count': data['Earn_Change']
        })
    if scores_list:
        correlation_scores[kw_type] = pd.DataFrame(scores_list).set_index('Intention')

print("\nRational Intention - Action Change Correlation:")
print(correlation_scores['rational'])
print("\nEmotional Intention - Action Change Correlation:")
print(correlation_scores['emotional'])

# 7. Generate Plots (Matrix Style for State Correlation)
print("\nGenerating state-based correlation matrix plots...")

action_labels = ['Move (True)', 'Move (False)', 'Earn (True)', 'Earn (False)']
# prob_cols defined earlier as prob_state_cols

for kw_type in ['rational', 'emotional']:
    if kw_type in state_correlation_scores and not state_correlation_scores[kw_type].empty:
        df_to_plot = state_correlation_scores[kw_type][prob_state_cols]
        intentions = df_to_plot.index.tolist()
        num_intentions = len(intentions)
        num_actions = len(action_labels)

        if num_intentions == 0 or num_actions == 0:
            print(f"Skipping matrix plot for {kw_type} intentions: No intentions or actions to plot.")
            continue

        fig, ax = plt.subplots(figsize=(8, max(4, num_intentions * 0.8))) # Adjust height

        # Create the grid and plot markers
        max_prob = df_to_plot.values.max() # Find max probability for scaling
        min_prob = df_to_plot.values.min()

        for i, intention in enumerate(intentions):
            for j, prob_col in enumerate(prob_state_cols):
                prob = df_to_plot.loc[intention, prob_col]
                # Scale marker size based on probability (adjust scaling factor as needed)
                # Avoid zero size markers, add a small base size
                marker_size = 50 + prob * 800 # Example scaling: base size 50, max additional 800
                ax.scatter(j, i, s=marker_size, alpha=0.7, cmap='viridis', c=[prob], vmin=min_prob, vmax=max_prob) # Use probability for color too
                # Add text annotation for probability
                ax.text(j, i, f'{prob:.2f}', ha='center', va='center', fontsize=9, color='white' if prob > 0.6 else 'black') # Adjust text color threshold

        # Set ticks and labels
        ax.set_xticks(np.arange(num_actions))
        ax.set_yticks(np.arange(num_intentions))
        ax.set_xticklabels(action_labels, rotation=45, ha='left') # Rotate labels slightly
        ax.set_yticklabels(intentions)

        # Adjust layout and appearance
        ax.set_xlim(-0.5, num_actions - 0.5)
        ax.set_ylim(num_intentions - 0.5, -0.5) # Correct inversion for scatter
        # ax.invert_yaxis() # Already handled by ylim
        ax.xaxis.tick_bottom() # Keep x-axis labels at bottom
        ax.xaxis.set_label_position('bottom')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.title(f'P(Action State | Top {TOP_N_INTENTIONS} {kw_type.capitalize()} Intention Presence)')
        # plt.xlabel('Action State') # X labels are clear enough
        # plt.ylabel('Intention Keyword') # Y labels are clear enough
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(RESULTS_DIR, f'{kw_type}_intention_action_state_correlation_matrix.png')
        try:
            plt.savefig(plot_filename)
            print(f"Saved matrix plot to: {plot_filename}")
        except Exception as e:
            print(f"Error saving matrix plot {plot_filename}: {e}")
        plt.close(fig)
    else:
        print(f"Skipping matrix plot for {kw_type} intentions: No correlation data calculated or available.")

# --- Optional: Keep or remove the old heatmap generation --- 
# print("Generating correlation heatmaps (change-based)...")
# for kw_type in ['rational', 'emotional']:
#     if kw_type in correlation_scores and not correlation_scores[kw_type].empty:
#         df_to_plot_heatmap = correlation_scores[kw_type][['Prob_Move_Change', 'Prob_Earn_Change']]
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(df_to_plot_heatmap, annot=True, cmap="viridis", fmt=".2f", linewidths=.5)
#         plt.title(f'Probability of Action Change Given Top {TOP_N_INTENTIONS} {kw_type.capitalize()} Intention Change')
#         plt.ylabel('Intention Keyword')
#         plt.xlabel('Action Change')
#         plt.xticks(rotation=0)
#         plt.yticks(rotation=0)
#         plt.tight_layout()
#         plot_filename_heatmap = os.path.join(RESULTS_DIR, f'{kw_type}_intention_action_CHANGE_correlation_heatmap.png')
#         try:
#             plt.savefig(plot_filename_heatmap)
#             print(f"Saved CHANGE heatmap to: {plot_filename_heatmap}")
#         except Exception as e:
#             print(f"Error saving CHANGE heatmap {plot_filename_heatmap}: {e}")
#         plt.close()
#     else:
#         print(f"Skipping CHANGE heatmap for {kw_type} intentions: No correlation data calculated.")

print("\nAnalysis complete.")
