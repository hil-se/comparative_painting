import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Directories
RESULT_DIR = "result"
OUTPUT_CSV_DIR = "averaged_results"
OUTPUT_PLOT_DIR = "plots_averaged"

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

def load_data_safe(filepath):
    """Safely load CSV file, return None if not found"""
    return pd.read_csv(filepath) if os.path.exists(filepath) else None


def average_rater_data(painting, rating_type, experiment_type):
    """
    Compute average rho and rs across all raters for a given experiment type.
    Returns the averaged DataFrames (reg_data_avg, comp_data_avg)
    """
    # File paths
    reg_file = f"{RESULT_DIR}/{painting}_{rating_type}_{experiment_type}.csv"
    comp_file = f"{RESULT_DIR}/{painting}_{rating_type}_{experiment_type}_comparative.csv"
    
    reg_data = load_data_safe(reg_file)
    comp_data = load_data_safe(comp_file)
    
    if reg_data is None or comp_data is None:
        print(f"  ⊘ Missing data for {painting} - {rating_type} - {experiment_type}")
        return None, None

    # --- Average comparative data (pairwise results) ---
    comp_avg = (
        comp_data.groupby("N")[["rho", "rs"]]
        .mean()
        .reset_index()
    )
    
    # --- Average regression data ---
    reg_avg = (
        reg_data[["rho", "rs"]].mean().to_frame().T
    )
    
    # Save averaged CSVs
    comp_out = f"{OUTPUT_CSV_DIR}/{painting}_{rating_type}_{experiment_type}_comparative_avg.csv"
    reg_out = f"{OUTPUT_CSV_DIR}/{painting}_{rating_type}_{experiment_type}_avg.csv"
    comp_avg.to_csv(comp_out, index=False)
    reg_avg.to_csv(reg_out, index=False)
    
    print(f"  ✓ Saved averaged CSVs: {experiment_type} ({painting} - {rating_type})")
    return reg_avg, comp_avg


def plot_experiment_average(painting, rating_type, experiment_type, reg_avg, comp_avg):
    """
    Plot a single averaged plot (ρ and rs vs N) for within/cross/average experiments.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'{experiment_type.title()} Comparison: {painting.title()} - {rating_type.title()}',
                 fontsize=16, fontweight='bold')

    # Plot pairwise results
    ax.plot(comp_avg['N'], comp_avg['rho'], marker='o', linewidth=2.5, markersize=8,
            label='Pairwise Pearson (ρ)', color='#FF8C00')
    
    ax.plot(comp_avg['N'], comp_avg['rs'], marker='s', linewidth=2.5, markersize=8,
            label='Pairwise Spearman (rs)', color='#FF6347')
    
    # Plot regression results (single mean lines)
    ax.axhline(y=reg_avg['rho'].iloc[0], color='#1E90FF', linestyle='--', linewidth=2.5,
               label='Regression Pearson (ρ)')
    ax.axhline(y=reg_avg['rs'].iloc[0], color='#4169E1', linestyle='--', linewidth=2.5,
               label='Regression Spearman (rs)')

    ax.set_xlabel('Sample Size (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.set_xticks(comp_avg['N'])

    all_vals = list(comp_avg['rho']) + list(comp_avg['rs']) + \
               [reg_avg['rho'].iloc[0], reg_avg['rs'].iloc[0]]
    y_min, y_max = min(all_vals) - 0.05, max(all_vals) + 0.05
    ax.set_ylim([max(-1, y_min), min(1, y_max)])
    
    plt.tight_layout()
    out_file = f"{OUTPUT_PLOT_DIR}/{painting}_{rating_type}_{experiment_type}_avg_plot.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot: {out_file}")
    plt.close()


def process_all_experiments():
    paintings = ["abstract", "representational"]
    rating_types = ["beauty", "liking"]
    experiment_types = ["average", "within", "cross"]
    
    print("\n" + "="*70)
    print("GENERATING AVERAGED EXPERIMENT RESULTS AND PLOTS")
    print("="*70)
    
    for painting in paintings:
        for rating_type in rating_types:
            print(f"\n{painting.upper()} - {rating_type.upper()}")
            print("-" * 50)
            for exp_type in experiment_types:
                reg_avg, comp_avg = average_rater_data(painting, rating_type, exp_type)
                if reg_avg is not None and comp_avg is not None:
                    plot_experiment_average(painting, rating_type, exp_type, reg_avg, comp_avg)
    
    print("\n" + "="*70)
    print("✓ All averaged plots and CSVs generated!")
    print(f"  CSVs → {OUTPUT_CSV_DIR}/")
    print(f"  Plots → {OUTPUT_PLOT_DIR}/")
    print("="*70)


if __name__ == "__main__":
    process_all_experiments()
