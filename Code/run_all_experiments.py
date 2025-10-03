import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# --- Main Imports ---
import data_preprocessor_resnet50 as data_loader
import regression_model
import pairwise_resnet
from regression_model import run_cross_rater_experiment
from pairwise_resnet import run_cross_rater_pairwise_experiment

# --- CONFIGURATION ---
MAIN_RESULTS_DIR = "Final_Analysis_Results"

# 1. CHOOSE THE EXPERIMENT MODE
# Options: 'AVERAGE', 'WITHIN_RATER', 'CROSS_RATER'
EXPERIMENT_MODE = 'AVERAGE' 

# 2. CHOOSE WHICH MODELS TO RUN (NEW!)
# Options: 'regression', 'pairwise', 'both'
MODELS_TO_RUN = 'both'

# 3. GENERAL SETTINGS
NUM_RUNS = 10
RATERS_TO_RUN = range(1, 6)
NUM_WORKERS = max(1, os.cpu_count() - 1)


# =================================================================================
# ===== WORKER FUNCTIONS FOR MULTIPROCESSING =====
# =================================================================================
def regression_worker(args):
    run_dir, preloaded_df = args
    regression_model.main(output_dir=run_dir, preloaded_df=preloaded_df)
    return True
def pairwise_worker(args):
    run_dir, n_value, preloaded_df = args
    pairwise_resnet.main(output_dir=run_dir, specific_n=n_value, preloaded_df=preloaded_df)
    return True
def cross_regression_worker(args):
    run_num, train_df, test_df, target, f_cols, cond_name = args
    result = run_cross_rater_experiment(train_df, test_df, target, f_cols, cond_name)
    if result: result['Run'] = run_num
    return result
def cross_pairwise_worker(args):
    run_num, n_value, train_df, test_df, target, f_cols, cond_name = args
    result = run_cross_rater_pairwise_experiment(train_df, test_df, target, n_value, f_cols, cond_name)
    if result: result['Run'] = run_num
    return result

# =================================================================================
# ===== PLOTTING FUNCTION (CORRECTED) =====
# =================================================================================

def plot_results(base_dir, title_prefix=""):
    print(f"\n--- Generating plots for: {title_prefix or 'Average Rater'} ---")
    reg_path = os.path.join(base_dir, "aggregated_regression_results.csv")
    pair_path = os.path.join(base_dir, "raw_pairwise_results.csv")

    plot_reg = os.path.exists(reg_path)
    plot_pair = os.path.exists(pair_path)

    if not plot_reg and not plot_pair:
        print(f"SKIPPING PLOTS: No result CSVs found in '{base_dir}'.")
        return

    plot_dir = os.path.join(base_dir, "comparison_plots")
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    reg_df = pd.read_csv(reg_path) if plot_reg else pd.DataFrame()
    pair_df_raw = pd.read_csv(pair_path) if plot_pair else pd.DataFrame()
    
    # Build the list of conditions intelligently based on available data
    condition_series = []
    if plot_reg:
        condition_series.append(reg_df['Condition'])
    if plot_pair:
        condition_series.append(pair_df_raw['Condition'])
    conditions = pd.concat(condition_series).unique()

    for condition in conditions:
        plt.figure(figsize=(10, 6.5))
        ax = plt.gca()
        
        if plot_pair:
            pair_df_raw_cond = pair_df_raw[pair_df_raw['Condition'] == condition]
            pair_df_raw_cond = pair_df_raw_cond.rename(columns={'Pearson_Corr': 'Pearson Correlation', 'Spearman_Corr': 'Spearman Correlation'})
            sns.lineplot(data=pair_df_raw_cond, x='n', y='Pearson Correlation', marker='o', label='Pairwise Pearson', ax=ax, errorbar='ci')
            sns.lineplot(data=pair_df_raw_cond, x='n', y='Spearman Correlation', marker='s', label='Pairwise Spearman', ax=ax, errorbar='ci')

        if plot_reg:
            reg_cond_df = reg_df[reg_df['Condition'] == condition]
            if not reg_cond_df.empty:
                ax.axhline(y=reg_cond_df['Pearson Correlation'].iloc[0], ls='--', color='blue', label='Regression Pearson Baseline')
                ax.axhline(y=reg_cond_df['Spearman Correlation'].iloc[0], ls='--', color='orange', label='Regression Spearman Baseline')
        
        ax.set_title(f"{title_prefix}{condition}", fontsize=16, pad=15)
        ax.set_xlabel("n (Pairs per Image)", fontsize=12)
        ax.set_ylabel("Correlation", fontsize=12)
        ax.set_ylim(bottom=0)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"comparison_{condition}.png"), dpi=150)
        plt.close()
    
    print(f"✅ Plots saved to: {plot_dir}")

# =================================================================================
# ===== EXPERIMENT RUNNER & AGGREGATION FUNCTIONS =====
# =================================================================================

def run_standard_experiments(base_dir, models_to_run, preloaded_df=None):
    if models_to_run in ['regression', 'both']:
        reg_base_dir = os.path.join(base_dir, "regression")
        tasks = [(os.path.join(reg_base_dir, f"run_{i}"), preloaded_df) for i in range(NUM_RUNS)]
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            list(tqdm(pool.imap_unordered(regression_worker, tasks), total=len(tasks), desc="Regression Runs"))

    if models_to_run in ['pairwise', 'both']:
        pair_base_dir = os.path.join(base_dir, "pairwise")
        tasks = [(os.path.join(pair_base_dir, f"n_{n}", f"run_{i}"), n, preloaded_df) for n in range(1, 16) for i in range(NUM_RUNS)]
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            list(tqdm(pool.imap_unordered(pairwise_worker, tasks), total=len(tasks), desc="Pairwise Runs"))

def aggregate_standard_results(base_dir, models_to_run):
    print("\n--- Aggregating Standard Results ---")
    if models_to_run in ['regression', 'both']:
        reg_base_dir = os.path.join(base_dir, "regression")
        all_reg_dfs = [pd.read_csv(os.path.join(reg_base_dir, f"run_{i}", "regression_results.csv")) for i in range(NUM_RUNS) if os.path.exists(os.path.join(reg_base_dir, f"run_{i}", "regression_results.csv"))]
        if all_reg_dfs:
            df_reg_avg = pd.concat(all_reg_dfs).groupby(['Condition']).mean(numeric_only=True).reset_index()
            df_reg_avg.to_csv(os.path.join(base_dir, "aggregated_regression_results.csv"), index=False)
        if os.path.exists(reg_base_dir): shutil.rmtree(reg_base_dir)

    if models_to_run in ['pairwise', 'both']:
        pair_base_dir = os.path.join(base_dir, "pairwise")
        all_pair_dfs = []
        for n in range(1, 16):
            for i in range(NUM_RUNS):
                f_path = os.path.join(pair_base_dir, f"n_{n}", f"run_{i}", "pairwise_results.csv")
                if os.path.exists(f_path):
                    df = pd.read_csv(f_path); df['n'] = n; df['run'] = i; all_pair_dfs.append(df)
        if all_pair_dfs:
            pd.concat(all_pair_dfs).to_csv(os.path.join(base_dir, "raw_pairwise_results.csv"), index=False)
        if os.path.exists(pair_base_dir): shutil.rmtree(pair_base_dir)
    print("Cleaned up intermediate run directories.")

def orchestrate_cross_rater_experiments(raters_to_process, models_to_run):
    for test_rater_id in raters_to_process:
        title = f"Cross-Rater (Test on Rater {test_rater_id})"
        print(f"\n{'='*25} PROCESSING: {title.upper()} {'='*25}")
        main_dir = os.path.join(MAIN_RESULTS_DIR, "Cross_Rater_Analysis", f"Rater_{test_rater_id}")
        if os.path.exists(main_dir): shutil.rmtree(main_dir)
        os.makedirs(main_dir, exist_ok=True)
        train_df = data_loader.load_and_preprocess_data(exclude_rater_id=test_rater_id)
        test_df = data_loader.load_and_preprocess_data(rater_id=test_rater_id)
        if train_df.empty or test_df.empty:
            print(f"Skipping rater {test_rater_id} due to empty dataset."); continue
        feature_cols = [c for c in train_df.columns if c.startswith(data_loader.FEATURE_PREFIX)]
        conditions = {'beauty_abstract': 'Beauty', 'liking_abstract': 'Liking', 'beauty_represent': 'Beauty', 'liking_represent': 'Liking'}
        if models_to_run in ['regression', 'both']:
            reg_tasks, all_reg_results = [], []
            for name, target in conditions.items():
                train_cond_df = train_df[(train_df['Representational'] == (1 if 'represent' in name else 0)) & (train_df[target].notna())]
                test_cond_df = test_df[(test_df['Representational'] == (1 if 'represent' in name else 0)) & (test_df[target].notna())]
                for i in range(NUM_RUNS):
                    reg_tasks.append((i, train_cond_df, test_cond_df, target, feature_cols, name))
            with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                for result in tqdm(pool.imap_unordered(cross_regression_worker, reg_tasks), total=len(reg_tasks), desc="Cross-Reg Runs"):
                    if result: all_reg_results.append(result)
            if all_reg_results:
                df_reg_avg = pd.DataFrame(all_reg_results).groupby(["Condition"]).mean(numeric_only=True).reset_index()
                df_reg_avg.to_csv(os.path.join(main_dir, "aggregated_regression_results.csv"), index=False)
        if models_to_run in ['pairwise', 'both']:
            pair_tasks, all_pair_results = [], []
            for name, target in conditions.items():
                train_cond_df = train_df[(train_df['Representational'] == (1 if 'represent' in name else 0)) & (train_df[target].notna())]
                test_cond_df = test_df[(test_df['Representational'] == (1 if 'represent' in name else 0)) & (test_df[target].notna())]
                for n in range(1, 16):
                    for i in range(NUM_RUNS):
                        pair_tasks.append((i, n, train_cond_df, test_cond_df, target, feature_cols, name))
            with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                for result in tqdm(pool.imap_unordered(cross_pairwise_worker, pair_tasks), total=len(pair_tasks), desc="Cross-Pair Runs"):
                    if result: all_pair_results.append(result)
            if all_pair_results:
                pd.concat([pd.DataFrame([res]) for res in all_pair_results]).to_csv(os.path.join(main_dir, "raw_pairwise_results.csv"), index=False)
        plot_results(main_dir, title_prefix=f"Cross-Rater (Test on Rater {test_rater_id}): ")

# ===========================================================================
# ===== MAIN EXECUTION BLOCK =====
# ===========================================================================

if __name__ == '__main__':
    os.makedirs(MAIN_RESULTS_DIR, exist_ok=True)
    if EXPERIMENT_MODE == 'AVERAGE':
        main_dir = os.path.join(MAIN_RESULTS_DIR, "Average_Rater")
        if os.path.exists(main_dir): shutil.rmtree(main_dir)
        avg_df = data_loader.load_and_preprocess_data()
        run_standard_experiments(main_dir, MODELS_TO_RUN, preloaded_df=avg_df)
        aggregate_standard_results(main_dir, MODELS_TO_RUN)
        plot_results(main_dir, title_prefix="Average Rater: ")
    elif EXPERIMENT_MODE == 'WITHIN_RATER':
        for rater_num in RATERS_TO_RUN:
            main_dir = os.path.join(MAIN_RESULTS_DIR, "Rater_{rater_num}")
            if os.path.exists(main_dir): shutil.rmtree(main_dir)
            rater_df = data_loader.load_and_preprocess_data(rater_id=rater_num)
            run_standard_experiments(main_dir, MODELS_TO_RUN, preloaded_df=rater_df)
            aggregate_standard_results(main_dir, MODELS_TO_RUN)
            plot_results(main_dir, title_prefix=f"Rater {rater_num}: ")
    elif EXPERIMENT_MODE == 'CROSS_RATER':
        orchestrate_cross_rater_experiments(raters_to_process=RATERS_TO_RUN, models_to_run=MODELS_TO_RUN)
    print("\n\nAll selected experiments and plotting complete!")