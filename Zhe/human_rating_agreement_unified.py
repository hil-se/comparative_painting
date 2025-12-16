import numpy as np
import pandas as pd
from collections import Counter


def generate_pairs_absolute(x1, x2):
    """Generate pairwise preferences from absolute ratings"""
    n = len(x1)
    pairs = {"A": [], "B": [], "agree": []}
    for i in range(n):
        for j in range(i+1, n):
            d1 = d2 = 0
            if x1[i] > x1[j]:
                d1 = 1
            elif x1[i] < x1[j]:
                d1 = -1
            if x2[i] > x2[j]:
                d2 = 1
            elif x2[i] < x2[j]:
                d2 = -1
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
    """Calculate accuracy and Cohen's kappa - works for both methods"""
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
        kappa = 1-(1-acc)/(1-pe)
    
    return acc, kappa


def create_gt_comparative_column(df):
    """
    Create a GT column for comparative data by comparing GT_A vs GT_B
    Returns 'A' if GT_A > GT_B, 'B' if GT_B > GT_A, None if equal
    """
    def compare_gt(row):
        if row['GT_A'] > row['GT_B']:
            return 'A'
        elif row['GT_B'] > row['GT_A']:
            return 'B'
        else:
            return None  # Tie case
    
    df['GT'] = df.apply(compare_gt, axis=1)
    return df


def analyze_ratings(input_file, output_file, rating_type='absolute', raters=None):
    """
    Analyze inter-rater agreement for either absolute or comparative ratings
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        rating_type: 'absolute' or 'comparative'
        raters: List of rater column names (default: all P1-P6 for comparative, P1,P3-P6 for absolute)
    """
    df = pd.read_csv(input_file)
    
    # For comparative data, create GT column from GT_A and GT_B if they exist
    if rating_type == 'comparative' and 'GT_A' in df.columns and 'GT_B' in df.columns:
        df = create_gt_comparative_column(df)
        # Remove rows where GT is None (ties)
        initial_rows = len(df)
        df = df[df['GT'].notna()]
        ties_removed = initial_rows - len(df)
        if ties_removed > 0:
            print(f"Note: Removed {ties_removed} tie(s) where GT_A == GT_B")
    
    # Set default raters based on type
    if raters is None:
        if rating_type == 'absolute':
            raters = ["GT", "P1", "P2", "P3", "P4", "P5", "P6"]  # P2 excluded
        else:
            # For comparative, include GT only if it exists in the dataframe
            if 'GT' in df.columns:
                raters = ["GT", "P1", "P2", "P3", "P4", "P5", "P6"]
            else:
                raters = ["P1", "P2", "P3", "P4", "P5", "P6"]
    
    results = []
    for i in range(len(raters)):
        for j in range(i+1, len(raters)):
            # Skip if rater column doesn't exist
            if raters[i] not in df.columns or raters[j] not in df.columns:
                continue
                
            # Generate pairs based on rating type
            if rating_type == 'absolute':
                pairs = generate_pairs_absolute(df[raters[i]], df[raters[j]])
            else:
                pairs = generate_pairs_comparative(df[raters[i]], df[raters[j]])
            
            # Calculate agreement metrics (same function for both types)
            acc, kappa = acc_kappa(pairs)
            
            result = {
                "Pair": raters[i] + "/" + raters[j], 
                "Acc": "%.2f" % acc, 
                "Kappa": "%.2f" % kappa
            }
            results.append(result)
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"\n{rating_type.upper()} - {input_file}")
    print(result_df)
    return result_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Define all analysis tasks
    analyses = [
        # Absolute ratings
        {
            'type': 'absolute',
            'input': 'human_survey_data/Abstract_Beauty_Absolute.csv',
            'output': 'human_survey_data/agreements_absolute/Agreement_Abstract_Beauty_Absolute.csv'
        },
        {
            'type': 'absolute',
            'input': 'human_survey_data/Abstract_Liking_Absolute.csv',
            'output': 'human_survey_data/agreements_absolute/Agreement_Abstract_Liking_Absolute.csv'
        },
        {
            'type': 'absolute',
            'input': 'human_survey_data/Repr_Beauty_Absolute.csv',
            'output': 'human_survey_data/agreements_absolute/Agreement_Repr_Beauty_Absolute.csv'
        },
        {
            'type': 'absolute',
            'input': 'human_survey_data/Repr_Liking_Absolute.csv',
            'output': 'human_survey_data/agreements_absolute/Agreement_Repr_Liking_Absolute.csv'
        },
        
        # Comparative ratings
        {
            'type': 'comparative',
            'input': 'human_survey_data/Abstract_Beauty_Comparative.csv',
            'output': 'human_survey_data/agreements_absolute/Agreement_Abstract_Beauty_Comparative.csv'
        },
        {
            'type': 'comparative',
            'input': 'human_survey_data/Abstract_Liking_Comparative.csv',
            'output': 'human_survey_data/agreements_absolute/Agreement_Abstract_Liking_Comparative.csv'
        },
        {
            'type': 'comparative',
            'input': 'human_survey_data/Repr_Beauty_Comparative.csv',
            'output': 'human_survey_data/agreements_absolute/Agreement_Repr_Beauty_Comparative.csv'
        },
        {
            'type': 'comparative',
            'input': 'human_survey_data/Repr_Liking_Comparative.csv',
            'output': 'human_survey_data/agreements_absolute/Agreement_Repr_Liking_Comparative.csv'
        }
    ]
    
    # Run all analyses
    print("="*80)
    print("INTER-RATER AGREEMENT ANALYSIS")
    print("="*80)
    
    for analysis in analyses:
        try:
            analyze_ratings(
                input_file=analysis['input'],
                output_file=analysis['output'],
                rating_type=analysis['type']
            )
        except Exception as e:
            print(f"Error processing {analysis['input']}: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)