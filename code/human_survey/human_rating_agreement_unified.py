import numpy as np
import pandas as pd
from collections import Counter
import os


def generate_pairs_absolute(x1, x2):
    """Generate pairwise preferences from absolute ratings"""
    n = len(x1)
    pairs = {"A": [], "B": [], "agree": []}
    for i in range(n):
        for j in range(i+1, n):
            d1 = d2 = 0
            if x1[i] > x1[j]: d1 = 1
            elif x1[i] < x1[j]: d1 = -1
            
            if x2[i] > x2[j]: d2 = 1
            elif x2[i] < x2[j]: d2 = -1
            
            if d1 != 0 and d2 != 0:
                pairs["A"].append(d1)
                pairs["B"].append(d2)
                pairs["agree"].append(d1 == d2)
    return pairs


def generate_pairs_comparative(x1, x2):
    """Generate agreement pairs from comparative responses (A/B choices)"""
    n = len(x1)
    pairs = {"A": [], "B": [], "agree": []}
    for i in range(n):
        choice1 = 1 if x1.iloc[i] == 'A' else -1
        choice2 = 1 if x2.iloc[i] == 'A' else -1
        pairs["A"].append(choice1)
        pairs["B"].append(choice2)
        pairs["agree"].append(choice1 == choice2)
    return pairs


def acc_kappa(pairs):
    """Calculate accuracy and Cohen's kappa"""
    n = len(pairs["agree"])
    if n == 0:
        return 0, 0
    
    acc = np.sum(pairs["agree"]) / n
    count_A = Counter(pairs["A"])
    count_B = Counter(pairs["B"])
    pe = n**(-2) * (count_A[1]*count_B[1] + count_A[-1]*count_B[-1])
    
    if pe == 1:
        kappa = 1.0
    else:
        kappa = (acc - pe) / (1 - pe)
    
    return acc, kappa


def create_gt_comparative_column(df):
    """Create a GT column for comparative data"""
    def compare_gt(row):
        if row['GT_A'] > row['GT_B']: return 'A'
        elif row['GT_B'] > row['GT_A']: return 'B'
        else: return None
    
    df['GT'] = df.apply(compare_gt, axis=1)
    return df


def analyze_ratings(input_file, output_file, summary_file, rating_type='absolute', raters=None):
    """
    Analyze inter-rater agreement AND calculate summaries in a single pass.
    """
    df = pd.read_csv(input_file)
    
    # Pre-processing for comparative GT
    if rating_type == 'comparative' and 'GT_A' in df.columns and 'GT_B' in df.columns:
        df = create_gt_comparative_column(df)
        df = df[df['GT'].notna()]
    
    # Set default raters
    if raters is None:
        if rating_type == 'absolute':
            raters = ["GT", "P1", "P2", "P3", "P4", "P5", "P6"]
        else:
            raters = ["GT", "P1", "P2", "P3", "P4", "P5", "P6"] if 'GT' in df.columns else ["P1", "P2", "P3", "P4", "P5", "P6"]

    # Initialize containers
    pair_results = []
    # Dictionary to hold running totals: { 'P1': {'acc': 0, 'kappa': 0, 'count': 0}, ... }
    rater_stats = {r: {'acc': 0.0, 'kappa': 0.0, 'count': 0} for r in raters}

    # --- SINGLE PASS LOOP (DRY Implementation) ---
    for i in range(len(raters)):
        for j in range(i+1, len(raters)):
            r1, r2 = raters[i], raters[j]
            
            # Skip missing columns
            if r1 not in df.columns or r2 not in df.columns:
                continue
                
            # 1. Generate Pairs
            if rating_type == 'absolute':
                pairs = generate_pairs_absolute(df[r1], df[r2])
            else:
                pairs = generate_pairs_comparative(df[r1], df[r2])
            
            # 2. Calculate Metrics
            acc, kappa = acc_kappa(pairs)
            
            # 3. Store Pairwise Result
            pair_results.append({
                "Pair": f"{r1}/{r2}", 
                "Acc": f"{acc:.2f}", 
                "Kappa": f"{kappa:.2f}"
            })

            # 4. Accumulate Summary Stats (The "Separate GT" Logic)
            # Logic: If one rater is GT, only GT tracks the stats. 
            #        If both are humans, both track the stats.
            
            # Handle Rater 1
            if r1 == 'GT':
                rater_stats[r1]['acc'] += acc
                rater_stats[r1]['kappa'] += kappa
                rater_stats[r1]['count'] += 1
                # Do NOT add to r2 (Human) stats because r1 is GT
            elif r2 == 'GT':
                # This case shouldn't theoretically happen if GT is index 0, but good for safety
                rater_stats[r2]['acc'] += acc
                rater_stats[r2]['kappa'] += kappa
                rater_stats[r2]['count'] += 1
                # Do NOT add to r1 (Human) stats
            else:
                # Both are humans
                rater_stats[r1]['acc'] += acc
                rater_stats[r1]['kappa'] += kappa
                rater_stats[r1]['count'] += 1
                
                rater_stats[r2]['acc'] += acc
                rater_stats[r2]['kappa'] += kappa
                rater_stats[r2]['count'] += 1

    # --- SAVE OUTPUTS ---
    
    # 1. Save Pairwise Results
    pd.DataFrame(pair_results).to_csv(output_file, index=False)
    
    # 2. Calculate and Save Summaries
    summary_data = []
    for rater, stats in sorted(rater_stats.items()):
        if stats['count'] > 0:
            summary_data.append({
                'Rater': rater,
                'Avg_Acc': f"{stats['acc'] / stats['count']:.3f}",
                'Avg_Kappa': f"{stats['kappa'] / stats['count']:.3f}",
                'Num_Pairs': stats['count']
            })
            
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n{rating_type.upper()} processed.")
    print(f"Pairs saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")
    
    return summary_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    SURVEY_DATA = "../../results/human_survey/survey_data"
    AGREE_ABS = "../../results/human_survey/agreement/absolute"
    AGREE_SUM = "../../results/human_survey/agreement/summary"
    os.makedirs(AGREE_ABS, exist_ok=True)
    os.makedirs(AGREE_SUM, exist_ok=True)

    analyses = [
        # Absolute
        {'type': 'absolute', 'input': f'{SURVEY_DATA}/Abstract_Beauty_Absolute.csv', 'name': 'Abstract Beauty (Abs)'},
        {'type': 'absolute', 'input': f'{SURVEY_DATA}/Abstract_Liking_Absolute.csv', 'name': 'Abstract Liking (Abs)'},
        {'type': 'absolute', 'input': f'{SURVEY_DATA}/Repr_Beauty_Absolute.csv', 'name': 'Repr Beauty (Abs)'},
        {'type': 'absolute', 'input': f'{SURVEY_DATA}/Repr_Liking_Absolute.csv', 'name': 'Repr Liking (Abs)'},
        # Comparative
        {'type': 'comparative', 'input': f'{SURVEY_DATA}/Abstract_Beauty_Comparative.csv', 'name': 'Abstract Beauty (Comp)'},
        {'type': 'comparative', 'input': f'{SURVEY_DATA}/Abstract_Liking_Comparative.csv', 'name': 'Abstract Liking (Comp)'},
        {'type': 'comparative', 'input': f'{SURVEY_DATA}/Repr_Beauty_Comparative.csv', 'name': 'Repr Beauty (Comp)'},
        {'type': 'comparative', 'input': f'{SURVEY_DATA}/Repr_Liking_Comparative.csv', 'name': 'Repr Liking (Comp)'}
    ]
    
    print("="*80)
    print("INTER-RATER AGREEMENT ANALYSIS (Single Pass)")
    print("="*80)
    
    for task in analyses:
        # Construct filenames dynamically
        base_name = os.path.basename(task['input']).replace('.csv', '')
        out_pair = f"{AGREE_ABS}/Agreement_{base_name}.csv"
        out_sum = f"{AGREE_SUM}/Summary_{base_name}.csv"
        
        try:
            analyze_ratings(
                input_file=task['input'],
                output_file=out_pair,
                summary_file=out_sum,
                rating_type=task['type']
            )
        except Exception as e:
            print(f"Error processing {task['name']}: {e}")
            
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE")
    print("="*80)