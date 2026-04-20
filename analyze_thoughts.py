import os
import json
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Added for quantile calculation
# Potential NLP libraries (install if needed)
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIRS = [f'exp{i}' for i in range(1, 6)]
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_results')
SUMMARY_FILE = os.path.join(OUTPUT_DIR, 'experiments_summary_corrected.csv')
TOP_N_KEYWORDS_HEATMAP = 15 # Number of top keywords for the heatmap
PERFORMANCE_METRIC_HEATMAP = 'individual_utility' # Metric to categorize for heatmap

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def extract_rider_id_from_filename(filename):
    match = re.match(r'(\d+)_thought.json', filename)
    return int(match.group(1)) if match else None

def load_thought_data(exp_dir_path):
    """Loads thought data for a single experiment."""
    thought_data = {}
    for filename in os.listdir(exp_dir_path):
        if filename.endswith('_thought.json'):
            rider_id = extract_rider_id_from_filename(filename)
            if rider_id is not None:
                file_path = os.path.join(exp_dir_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        think_text = ""
                        # Check if data is a list and not empty
                        if isinstance(data, list) and data:
                            # Assume the relevant info is in the first dictionary of the list
                            first_item = data[0]
                            if isinstance(first_item, dict):
                                # Prioritize 'think', then 'mixed_thought', then 'thought'
                                think_text = first_item.get('think') or first_item.get('mixed_thought') or first_item.get('thought') or ""
                            else:
                                print(f"Warning: First item in list is not a dict in {filename}. Skipping.")
                        # Check if data is a dictionary (original expected format)
                        elif isinstance(data, dict):
                            think_text = data.get('think') or data.get('mixed_thought') or data.get('thought') or ""
                        # Handle other unexpected data types
                        else:
                            print(f"Warning: Unexpected data type ({type(data)}) in {filename}, expected list or dict. Skipping.")
                        thought_data[rider_id] = think_text
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {filename}")
                except Exception as e:
                    print(f"Warning: Error reading {filename}: {e}")
    return thought_data

def preprocess_text(text):
    """Basic text preprocessing: lowercasing, removing punctuation (simple version)."""
    if not isinstance(text, str):
        return [] # Return empty list if input is not a string
    text = text.lower()
    text = re.sub(r'[\n\t,.!?"\'():;-]', ' ', text) # Remove common punctuation and newlines
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space
    # Add more sophisticated preprocessing (stopwords, stemming/lemmatization) if needed
    # Example with NLTK:
    # stop_words = set(stopwords.words('english')) | set(stopwords.words('chinese')) # Adjust languages
    # word_tokens = word_tokenize(text)
    # filtered_text = [w for w in word_tokens if not w.lower() in stop_words and w.isalpha()]
    # return filtered_text
    return text.split()

def extract_keywords(text, top_n=10):
    """Extracts top N keywords based on frequency after basic preprocessing."""
    words = preprocess_text(text)
    if not words:
        return []
    word_counts = Counter(words)
    # Filter out very short words or potentially irrelevant frequent words if needed
    # e.g., filter words shorter than 3 characters
    filtered_counts = {word: count for word, count in word_counts.items() if len(word) > 2}
    # Get the most common keywords
    keywords = [word for word, count in Counter(filtered_counts).most_common(top_n)]
    return keywords

def categorize_individual_performance(utility, q1, q3):
    """Categorizes individual performance based on quantiles."""
    if pd.isna(utility):
        return 'Unknown'
    if utility >= q3:
        return 'High'
    elif utility >= q1:
        return 'Medium'
    else:
        return 'Low'

# --- Main Analysis Logic ---
all_keywords = {}
all_thoughts = {}

print("Loading thought data...")
for exp_name in EXPERIMENT_DIRS:
    exp_dir_path = os.path.join(BASE_DIR, exp_name)
    if os.path.isdir(exp_dir_path):
        print(f"Processing {exp_name}...")
        thoughts = load_thought_data(exp_dir_path)
        all_thoughts[exp_name] = thoughts
        exp_keywords = {}
        for rider_id, thought_text in thoughts.items():
            if thought_text:
                 exp_keywords[rider_id] = extract_keywords(thought_text)
            else:
                 exp_keywords[rider_id] = []
        all_keywords[exp_name] = exp_keywords
    else:
        print(f"Warning: Directory not found - {exp_dir_path}")

print("\nLoading performance data...")
all_rider_details = []
for exp_name in EXPERIMENT_DIRS:
    # Adjusted path finding logic to be more robust
    details_file_options = [
        os.path.join(OUTPUT_DIR, exp_name, f"{exp_name}_rider_details.csv"), # analysis_results/expN/expN_rider_details.csv
        os.path.join(OUTPUT_DIR, f"{exp_name}_rider_details.csv")      # analysis_results/expN_rider_details.csv
    ]
    details_file = None
    for option in details_file_options:
        if os.path.exists(option):
            details_file = option
            break

    if details_file is None:
        print(f"Warning: Rider details file not found for {exp_name} in expected locations.")
        continue

    try:
        exp_df = pd.read_csv(details_file)
        exp_df['Experiment'] = exp_name
        # Rename 'rider_id' to 'Rider ID' if necessary to match keyword_df
        if 'rider_id' in exp_df.columns and 'Rider ID' not in exp_df.columns:
            exp_df = exp_df.rename(columns={'rider_id': 'Rider ID'})
        all_rider_details.append(exp_df)
    except FileNotFoundError:
        print(f"Error: Rider details file not found at {details_file}") # Should not happen with check above, but keep for safety
    except Exception as e:
        print(f"Error loading {details_file}: {e}")

if not all_rider_details:
    print("Error: No rider details data loaded. Exiting.")
    exit()

summary_df = pd.concat(all_rider_details, ignore_index=True)

# Verify required columns for merging and categorization
required_cols = ['Experiment', 'Rider ID', PERFORMANCE_METRIC_HEATMAP]
if not all(col in summary_df.columns for col in required_cols):
    print(f"Error: Failed to create summary_df with required columns: {required_cols}.")
    print("Columns found:", summary_df.columns)
    exit()

# --- Categorize Performance --- #
print(f"\nCategorizing performance based on '{PERFORMANCE_METRIC_HEATMAP}'...")
# Calculate quantiles across all riders
q1 = summary_df[PERFORMANCE_METRIC_HEATMAP].quantile(0.33)
q3 = summary_df[PERFORMANCE_METRIC_HEATMAP].quantile(0.66)
print(f"Performance Thresholds ({PERFORMANCE_METRIC_HEATMAP}): Low < {q1:.2f} <= Medium < {q3:.2f} <= High")
summary_df['Performance Category'] = summary_df[PERFORMANCE_METRIC_HEATMAP].apply(
    lambda x: categorize_individual_performance(x, q1, q3)
)

# --- Merge Data and Analyze Keyword-Performance Relationship --- #
# Create a structure to hold keyword presence per rider across experiments
rider_keyword_data = []
for exp_name, riders_keywords in all_keywords.items():
    for rider_id, keywords in riders_keywords.items():
        # Add a row for each keyword associated with the rider in that experiment
        # If no keywords, still add a row to potentially join later
        if keywords:
            for keyword in keywords:
                rider_keyword_data.append({
                    'Experiment': exp_name,
                    'Rider ID': rider_id,
                    'Keyword': keyword,
                    'HasKeyword': True
                })
        else:
             rider_keyword_data.append({
                'Experiment': exp_name,
                'Rider ID': rider_id,
                'Keyword': None, # Or a placeholder like 'NO_KEYWORDS'
                'HasKeyword': False
            })

keyword_df = pd.DataFrame(rider_keyword_data)

# Perform the merge
# Use a left merge to keep all performance data, matching keywords where available
merged_df = pd.merge(summary_df, keyword_df, on=['Experiment', 'Rider ID'], how='left')

# Fill NaN for 'HasKeyword' for riders who might be in summary but not have thought files/keywords
merged_df['HasKeyword'].fillna(False, inplace=True)
merged_df['Keyword'].fillna('NO_KEYWORDS', inplace=True) # Use a placeholder

print("\nAnalyzing keyword-performance relationship...")

# Calculate overall keyword counts from the merged data (considering multiple entries per rider if they used multiple keywords)
overall_keyword_list = merged_df[merged_df['HasKeyword']]['Keyword'].tolist()
if overall_keyword_list:
    overall_keyword_counts = Counter(overall_keyword_list)
    print("\nTop 20 Keywords Overall (from merged data):")
    top_20_keywords = [kw for kw, count in overall_keyword_counts.most_common(20)]
    for keyword in top_20_keywords:
        print(f"- {keyword}: {overall_keyword_counts[keyword]}")

    # Analyze performance for top keywords
    # Choose performance metrics to analyze
    performance_metrics = ['individual_utility', 'efficiency', 'revenue', 'cost'] # Use metrics from rider_details.csv

    print("\nAverage Performance Metrics for Top 10 Keywords:")
    print(f"{'Keyword':<15} | {'Count':<7} | {'Metric':<20} | {'Avg Value (Keyword Users)':<25} | {'Avg Value (Overall)':<20}")
    print("-" * 100)

    # Calculate overall average performance from the combined rider details
    overall_avg_performance = summary_df[performance_metrics].mean()

    for keyword in top_20_keywords[:10]: # Analyze top 10
        # Get unique rider-experiment pairs that used this keyword
        riders_with_keyword = merged_df[merged_df['Keyword'] == keyword][['Experiment', 'Rider ID']].drop_duplicates()
        # Filter the original summary_df to get performance of these specific riders
        keyword_user_performance = pd.merge(summary_df, riders_with_keyword, on=['Experiment', 'Rider ID'], how='inner')

        if not keyword_user_performance.empty:
            avg_perf_keyword = keyword_user_performance[performance_metrics].mean()
            count = overall_keyword_counts[keyword]
            for metric in performance_metrics:
                if metric in avg_perf_keyword and metric in overall_avg_performance:
                    print(f"{keyword:<15} | {count:<7} | {metric:<20} | {avg_perf_keyword[metric]:<25.4f} | {overall_avg_performance[metric]:<20.4f}")
                else:
                    print(f"{keyword:<15} | {count:<7} | {metric:<20} | {'Metric not found':<25} | {'Metric not found':<20}")
            print("-" * 100)
        else:
             print(f"{keyword:<15} | {overall_keyword_counts[keyword]:<7} | {'No performance data found':<20} | {'N/A':<25} | {'N/A':<20}")
             print("-" * 100)

else:
    print("No keywords extracted from thought files.")

# --- Visualization --- #
# Visualization 1: Top Keyword Frequency (Improved)
if overall_keyword_list:
    top_15_keywords_freq = overall_keyword_counts.most_common(15)
    keywords_freq, counts_freq = zip(*top_15_keywords_freq)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(counts_freq), y=list(keywords_freq), palette='viridis')
    plt.title('Top 15 Most Frequent Keywords in Rider Thoughts (Overall)')
    plt.xlabel('Total Occurrences (Across all Riders/Experiments)')
    plt.ylabel('Keyword')
    plt.tight_layout()
    plot_path_freq = os.path.join(OUTPUT_DIR, 'top_keywords_frequency.png')
    plt.savefig(plot_path_freq)
    print(f"\nSaved keyword frequency plot to: {plot_path_freq}")
    plt.close()
else:
    print("Skipping keyword frequency visualization as no keywords were found.")

# Visualization 2: Performance Comparison for Top Keywords (Example: individual_utility)
if overall_keyword_list and not overall_avg_performance.empty and 'individual_utility' in overall_avg_performance.index:
    metric_to_plot = 'individual_utility'
    top_10_kw_perf = []
    keywords_for_plot = []
    for keyword in top_20_keywords[:10]:
        riders_with_keyword = merged_df[merged_df['Keyword'] == keyword][['Experiment', 'Rider ID']].drop_duplicates()
        keyword_user_performance = pd.merge(summary_df, riders_with_keyword, on=['Experiment', 'Rider ID'], how='inner')
        if not keyword_user_performance.empty and metric_to_plot in keyword_user_performance.columns:
            avg_perf = keyword_user_performance[metric_to_plot].mean()
            top_10_kw_perf.append(avg_perf)
            keywords_for_plot.append(keyword)

    if keywords_for_plot:
        plt.figure(figsize=(12, 7))
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({'Keyword': keywords_for_plot, f'Avg {metric_to_plot}': top_10_kw_perf})
        plot_df = plot_df.sort_values(f'Avg {metric_to_plot}', ascending=False)

        sns.barplot(x=f'Avg {metric_to_plot}', y='Keyword', data=plot_df, palette='coolwarm')
        # Add a line for the overall average
        plt.axvline(x=overall_avg_performance[metric_to_plot], color='grey', linestyle='--', label=f'Overall Avg ({overall_avg_performance[metric_to_plot]:.2f})')
        plt.title(f'Average {metric_to_plot} for Riders Using Top 10 Keywords')
        plt.xlabel(f'Average {metric_to_plot}')
        plt.ylabel('Keyword')
        plt.legend()
        plt.tight_layout()
        plot_path_perf = os.path.join(OUTPUT_DIR, 'keyword_performance_comparison.png')
        plt.savefig(plot_path_perf)
        print(f"Saved keyword performance comparison plot to: {plot_path_perf}")
        plt.close()
    else:
        print(f"Skipping performance comparison plot: Not enough data for {metric_to_plot}.")
else:
    print(f"Skipping performance comparison plot: No keywords or '{metric_to_plot}' not found in performance data.")

# Visualization 3: Keyword vs Performance Category Heatmap (Matrix-like)
if overall_keyword_list:
    print(f"\nGenerating heatmap for Top {TOP_N_KEYWORDS_HEATMAP} keywords vs Performance Category...")
    # Get the actual top N keywords based on overall frequency
    top_n_keywords_list = [kw for kw, count in overall_keyword_counts.most_common(TOP_N_KEYWORDS_HEATMAP)]

    # Filter the merged data for these keywords and valid performance categories
    heatmap_data = merged_df[
        merged_df['Keyword'].isin(top_n_keywords_list) &
        merged_df['Performance Category'].isin(['Low', 'Medium', 'High'])
    ]

    if not heatmap_data.empty:
        # Create the pivot table: Count riders per Keyword per Performance Category
        # We need unique rider-experiment pairs for the count
        unique_rider_exp_keyword = heatmap_data[['Experiment', 'Rider ID', 'Keyword', 'Performance Category']].drop_duplicates()

        pivot_table = unique_rider_exp_keyword.groupby(['Keyword', 'Performance Category']).size().unstack(fill_value=0)

        # Ensure all categories are present and ordered
        pivot_table = pivot_table.reindex(columns=['Low', 'Medium', 'High'], fill_value=0)
        # Order rows by keyword frequency (optional, but often helpful)
        pivot_table = pivot_table.reindex(top_n_keywords_list, fill_value=0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt="d", cmap="viridis", linewidths=.5)
        plt.title(f'Heatmap: Top {TOP_N_KEYWORDS_HEATMAP} Keywords vs. Rider Performance Category\n(Based on {PERFORMANCE_METRIC_HEATMAP})')
        plt.xlabel('Performance Category')
        plt.ylabel('Keyword')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()

        heatmap_plot_path = os.path.join(OUTPUT_DIR, 'keyword_performance_heatmap.png')
        try:
            plt.savefig(heatmap_plot_path)
            print(f"Saved keyword-performance heatmap to: {heatmap_plot_path}")
        except Exception as e:
            print(f"Error saving heatmap: {e}")
        plt.close()
    else:
        print("Skipping heatmap: No data available for the top keywords and performance categories.")
else:
    print("Skipping heatmap: No keywords found.")


# --- Discussion on Grid Visualization (User's Example) ---
# Creating a visualization exactly like the user's example (grid with matches)
# presents challenges with the current data:
# 1. Mapping Keywords to Grid: How should keywords be positioned on a 2D grid?
#    Requires categorization or embedding (e.g., semantic similarity).
# 2. Defining "Match": What constitutes a 'match' between a keyword (App) and
#    rider performance (User)? Is it simply using the keyword? Or using it AND
#    having high performance?
# 3. Defining Quadrants: The example mentions a top-right 'undesirable' quadrant.
#    This implies axes with meaning (e.g., X=Keyword Impact, Y=Performance Level).
#    Defining these axes and the 'undesirable' threshold needs clear criteria.
#
# The generated heatmap provides a matrix-like view relating keyword presence to performance categories,
# which is a step towards the user's request using the available data.
#
# Possible Approaches (More Complex):
# - Use NLP embeddings (like Word2Vec, BERT) to get vector representations of thoughts/keywords.
# - Use dimensionality reduction (PCA, t-SNE) to plot these embeddings in 2D.
# - Color points based on performance metrics.
# - Cluster keywords based on semantic meaning or performance correlation.
# - This would allow placing related keywords together on a grid/plot, potentially
#   revealing patterns closer to the user's abstract example.
#
# For now, the bar charts and heatmap provide direct analysis of frequency and average performance correlation.


# --- Further Analysis Ideas ---
# - Correlate keyword presence with specific performance metrics from summary_df.
# - Compare keyword usage between high-performing and low-performing riders.
# - Analyze how keyword usage changes across experiments (exp1 vs exp5).
# - Use more sophisticated NLP (TF-IDF, embeddings) for keyword extraction and analysis.
# - Create a visualization closer to the user's example (e.g., a heatmap showing keyword presence vs. performance groups).

print("\nThought analysis script finished.")