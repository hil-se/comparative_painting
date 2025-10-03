import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def cronbach_alpha(df):
    """
    Calculates Cronbach's Alpha for inter-rater reliability.
    Assumes df is a (items x raters) DataFrame.
    """
    df = df.dropna()
    k = df.shape[1]
    if k < 2: return np.nan
    
    item_vars = df.var(axis=1, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

def perform_advanced_analysis():
    """
    Loads rater data and performs an in-depth analysis to identify noise and patterns.
    """
    # --- 1. Load and Prepare Data ---
    print("--- 1. Loading and Preparing Data ---")
    try:
        df_abs_beauty = pd.read_csv('Abstract_All_Raters.csv')
        df_abs_liking = pd.read_csv('Abstract_Liking_All_Raters.csv')
        df_rep_beauty = pd.read_csv('Representational_All_Raters.csv')
        df_rep_liking = pd.read_csv('Representational_Liking_All_Raters.csv')
        print("✅ Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure this script is in the same folder as your CSV files.")
        return

    df_abs_beauty['Representational'] = 0; df_rep_beauty['Representational'] = 1
    df_abs_liking['Representational'] = 0; df_rep_liking['Representational'] = 1
    df_beauty = pd.concat([df_abs_beauty, df_rep_beauty])
    df_liking = pd.concat([df_abs_liking, df_rep_liking])
    df_full = pd.merge(df_beauty, df_liking, on=['Painting', 'Rater', 'Representational'], how='outer')
    print("-" * 40 + "\n")

    # --- 2. Inter-Rater Reliability Analysis ---
    print("--- 2. Inter-Rater Reliability Analysis ---")
    beauty_pivot = df_full.pivot_table(index='Painting', columns='Rater', values='Beauty')
    liking_pivot = df_full.pivot_table(index='Painting', columns='Rater', values='Liking')
    
    beauty_corr = beauty_pivot.corr()
    liking_corr = liking_pivot.corr()

    # Calculate average correlation for each rater
    avg_corr_beauty = beauty_corr.apply(lambda x: x.drop(x.name).mean()).sort_values()
    avg_corr_liking = liking_corr.apply(lambda x: x.drop(x.name).mean()).sort_values()
    
    print("\n>> Average Correlation of Each Rater with the Group (Beauty):")
    print(avg_corr_beauty.round(3))
    print("\n>> Average Correlation of Each Rater with the Group (Liking):")
    print(avg_corr_liking.round(3))

    # Calculate Cronbach's Alpha
    alpha_beauty = cronbach_alpha(beauty_pivot)
    alpha_liking = cronbach_alpha(liking_pivot)
    print(f"\n>> Cronbach's Alpha for Beauty Ratings: {alpha_beauty:.3f}")
    print(f">> Cronbach's Alpha for Liking Ratings: {alpha_liking:.3f}")
    print("(Note: Alpha > 0.7 is generally considered acceptable reliability)")
    print("-" * 40 + "\n")

    # --- 3. Outlier and Noise Detection ---
    print("--- 3. Outlier and Noise Detection ---")
    # Calculate consensus (mean) and disagreement (std) for each painting
    painting_stats = df_full.groupby('Painting').agg(
        Beauty_Mean=('Beauty', 'mean'), Beauty_Std=('Beauty', 'std'),
        Liking_Mean=('Liking', 'mean'), Liking_Std=('Liking', 'std')
    ).reset_index()

    df_full = pd.merge(df_full, painting_stats, on='Painting')

    # Calculate Z-score for each rating
    df_full['Beauty_ZScore'] = (df_full['Beauty'] - df_full['Beauty_Mean']) / df_full['Beauty_Std']
    df_full['Liking_ZScore'] = (df_full['Liking'] - df_full['Liking_Mean']) / df_full['Liking_Std']
    df_full['Abs_Beauty_ZScore'] = df_full['Beauty_ZScore'].abs()
    df_full['Abs_Liking_ZScore'] = df_full['Liking_ZScore'].abs()

    print("\n>> Top 15 Most Controversial Paintings (Highest Disagreement):")
    print(painting_stats.sort_values(by='Beauty_Std', ascending=False).head(15).round(2))
    
    print("\n>> Top 20 Most Outlying 'Beauty' Ratings (Potential Noise):")
    print(df_full.sort_values(by='Abs_Beauty_ZScore', ascending=False)[['Painting', 'Rater', 'Beauty', 'Beauty_Mean', 'Beauty_ZScore']].head(20).round(2))
    
    print("\n>> Top 20 Most Outlying 'Liking' Ratings (Potential Noise):")
    print(df_full.sort_values(by='Abs_Liking_ZScore', ascending=False)[['Painting', 'Rater', 'Liking', 'Liking_Mean', 'Liking_ZScore']].head(20).round(2))
    print("-" * 40 + "\n")
    
    # --- 4. Visualization ---
    print("--- 4. Generating Diagnostic Plots ---")
    output_dir = "eda_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Average Correlation per Rater
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    avg_corr_beauty.plot(kind='barh', color='skyblue')
    plt.title('Average Correlation per Rater (Beauty)')
    plt.xlabel('Average Pearson Correlation')
    
    plt.subplot(1, 2, 2)
    avg_corr_liking.plot(kind='barh', color='salmon')
    plt.title('Average Correlation per Rater (Liking)')
    plt.xlabel('Average Pearson Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_correlation_per_rater.png'))
    plt.close()
    
    # Plot 2: Z-Score Distributions
    plt.figure(figsize=(15, 6))
    sns.histplot(df_full['Beauty_ZScore'].dropna(), bins=50, kde=True, color='skyblue', label='Beauty Z-Scores')
    sns.histplot(df_full['Liking_ZScore'].dropna(), bins=50, kde=True, color='salmon', label='Liking Z-Scores')
    plt.title('Distribution of Rating Z-Scores')
    plt.xlabel('Z-Score (Deviation from Painting Mean)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'zscore_distribution.png'))
    plt.close()

    print(f"✅ Diagnostic plots saved to '{output_dir}' directory.")
    print("-" * 40 + "\n")

if __name__ == '__main__':
    perform_advanced_analysis()