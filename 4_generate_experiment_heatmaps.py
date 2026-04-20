import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor # Still useful for initial data loading if very many files

# Define the base directory and output directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_results', 'gridded_experiment_heatmaps')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the experiments
EXPERIMENT_DIRS = [f'exp{i}' for i in range(1, 6)]

# --- Configuration for Gridded Heatmaps ---
N_BINS = 10  # Number of bins for x and y axes (e.g., 20x20 grid, 30x30 grid). Adjust for desired granularity.
COLOR_MAP = "viridis"  # Colormap for the heatmaps. Try "plasma", "magma", "cividis", "Greens", "Blues", "BuPu"
# For example, if you want a style similar to the green part of the provided image: COLOR_MAP = "Greens"

def get_experiment_data(exp_name):
    """
    Loads and preprocesses data for ALL riders in a given experiment.
    Scales coordinates and returns a single DataFrame with 'x', 'y'.
    """
    exp_dir_path = os.path.join(BASE_DIR, exp_name)
    data_files = [f for f in os.listdir(exp_dir_path) if f.startswith('deliver') and f.endswith('_record.csv')]
    
    if not data_files:
        print(f"No data files found for experiment {exp_name}")
        return pd.DataFrame(columns=['x', 'y'])

    all_data_frames_exp = []
    for file_name in data_files:
        file_path = os.path.join(exp_dir_path, file_name)
        try:
            df = pd.read_csv(file_path)
            if 'x' not in df.columns or 'y' not in df.columns:
                print(f"Warning: 'x' or 'y' column missing in {file_path}. Skipping.")
                continue
            
            df['x'] = pd.to_numeric(df['x'], errors='coerce') / 20
            df['y'] = pd.to_numeric(df['y'], errors='coerce') / 20
            all_data_frames_exp.append(df[['x', 'y']]) # Only keep necessary columns
        except pd.errors.EmptyDataError:
            print(f"Warning: File {file_path} is empty. Skipping.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_data_frames_exp:
        print(f"No valid data loaded for experiment {exp_name}")
        return pd.DataFrame(columns=['x', 'y'])

    full_exp_data_df = pd.concat(all_data_frames_exp, ignore_index=True)
    full_exp_data_df.dropna(subset=['x', 'y'], inplace=True)
    return full_exp_data_df

def generate_faceted_gridded_heatmaps():
    print("Starting generation of faceted gridded heatmaps...")

    # 1. Load all data to determine global x and y ranges for consistent binning
    all_dfs = []
    with ProcessPoolExecutor() as executor: # Using ProcessPoolExecutor for potentially faster I/O
        results = list(executor.map(get_experiment_data, EXPERIMENT_DIRS))

    for df_exp in results:
        if not df_exp.empty:
            all_dfs.append(df_exp)

    if not all_dfs:
        print("No data found across all experiments. Cannot generate heatmaps.")
        return

    combined_all_data = pd.concat(all_dfs, ignore_index=True)
    if combined_all_data.empty:
        print("Combined data is empty after loading all experiments.")
        return

    # Define ranges for histogram2d, possibly with a small padding or based on actual data range
    global_x_min = combined_all_data['x'].min()
    global_x_max = combined_all_data['x'].max()
    global_y_min = combined_all_data['y'].min()
    global_y_max = combined_all_data['y'].max()

    # Ensure ranges are valid (max > min)
    if global_x_min == global_x_max: global_x_max += 1e-6 # Add small epsilon if all values are same
    if global_y_min == global_y_max: global_y_max += 1e-6

    # Define bin edges based on global ranges
    # np.linspace creates N_BINS+1 edges for N_BINS bins
    x_bin_edges = np.linspace(global_x_min, global_x_max, N_BINS + 1)
    y_bin_edges = np.linspace(global_y_min, global_y_max, N_BINS + 1)

    # 2. Prepare heatmap data for each experiment and find the global max count for color scaling
    heatmap_counts_list = []
    max_count_across_all = 0

    for i, exp_name in enumerate(EXPERIMENT_DIRS):
        df_exp = results[i] # Get the pre-loaded data
        if df_exp.empty or df_exp[['x', 'y']].isnull().all().all():
            print(f"No valid data for gridded heatmap in {exp_name}")
            heatmap_counts_list.append(np.zeros((N_BINS, N_BINS))) # Placeholder for empty plot
            continue

        # Create 2D histogram for the current experiment
        # Note: np.histogram2d expects (sample_x, sample_y, bins=(xbins, ybins))
        # or (sample_y, sample_x, bins=(ybins, xbins)) if you prefer y first.
        # For sns.heatmap, rows are typically y and columns are x.
        # So, if hist_y is first dim of counts, it's rows.
        counts, _, _ = np.histogram2d(df_exp['x'].values, df_exp['y'].values, bins=[x_bin_edges, y_bin_edges])
        
        # Transpose 'counts' because heatmap expects rows as the first dimension of bins (y-axis),
        # and columns as the second dimension of bins (x-axis).
        # np.histogram2d with bins=[x_edges, y_edges] means counts[ix, iy].
        # To have y as rows and x as columns for heatmap:
        # Either use np.histogram2d(df_exp['y'], df_exp['x'], bins=[y_bin_edges, x_bin_edges])
        # and use counts directly, or transpose if you used (x,y) with (x_edges,y_edges)
        counts = counts.T # Transpose so that rows correspond to y-bins, columns to x-bins

        heatmap_counts_list.append(counts)
        if counts.size > 0:
            current_max = np.max(counts)
            if current_max > max_count_across_all:
                max_count_across_all = current_max
    
    if max_count_across_all == 0:
        print("All heatmap counts are zero. Plots will be blank.")
        max_count_across_all = 1 # Avoid vmin=vmax=0 for colorbar

    # 3. Create subplots and plot heatmaps
    nrows, ncols = 2, 3  # For 5 experiments, 2 rows, 3 columns is good.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4.5, nrows * 4), constrained_layout=False) # Adjust figsize
    axes_flat = axes.flatten()

    # Create a ScalarMappable for the global colorbar
    norm = plt.Normalize(vmin=0, vmax=max_count_across_all)
    sm = plt.cm.ScalarMappable(cmap=COLOR_MAP, norm=norm)
    sm.set_array([]) # Dummy data for the mappable

    for i, exp_name in enumerate(EXPERIMENT_DIRS):
        ax = axes_flat[i]
        counts = heatmap_counts_list[i]

        if counts is None or np.sum(counts) == 0 and not (counts.shape[0] == N_BINS and counts.shape[1] == N_BINS) :
             ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=9)
             ax.set_title(f'{exp_name}\n(No Data)', fontsize=10)
        else:
            # Use origin='lower' if you want (0,0) of your data to be at the bottom-left of the heatmap cell grid.
            # Default is 'upper'. Given typical (x,y) coordinate systems, 'lower' is often more intuitive
            # if y_bin_edges are increasing.
            sns.heatmap(counts, ax=ax, cmap=COLOR_MAP,
                        vmin=0, vmax=max_count_across_all,
                        cbar=False, # We'll use a global colorbar
                        square=False, # Let aspect be controlled by subplot; use ax.set_aspect('equal') if needed after
                        linewidths=.5, linecolor='lightgray', # Grid lines
                        xticklabels=False, yticklabels=False,
                        annot=False
                       )
            # To make heatmap cells truly square if N_BINS_X != N_BINS_Y or subplot aspect is not square:
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'{exp_name}', fontsize=10)

        # Remove axis spines for a cleaner look if desired
        # sns.despine(ax=ax, left=True, bottom=True)


    # Hide any unused subplots
    for j in range(len(EXPERIMENT_DIRS), nrows * ncols):
        axes_flat[j].axis('off')

    # Add a global colorbar
    # Adjust rect values: [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7]) # Position for a single shared colorbar
    fig.colorbar(sm, cax=cbar_ax, label='Activity Count')

    fig.suptitle('Gridded Rider Activity Heatmaps per Experiment', fontsize=16, y=0.98)
    
    # Adjust layout to prevent overlap and make space for title & colorbar
    # plt.tight_layout(rect=[0, 0, 0.9, 0.95]) # rect can conflict with add_axes sometimes
    plt.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.05, wspace=0.25, hspace=0.35)


    output_path = os.path.join(OUTPUT_DIR, f'faceted_gridded_heatmaps_nbins{N_BINS}.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig) # Close the figure to free memory
    print(f"Faceted gridded heatmap saved at {output_path}")

if __name__ == "__main__":
    generate_faceted_gridded_heatmaps()
