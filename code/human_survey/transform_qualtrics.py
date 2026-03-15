
"""
FINAL TRANSFORMATION SCRIPT
Aesthetic Judgment Study - Raw Qualtrics to Clean CSV

This script:
1. Loads your raw Qualtrics export
2. Filters for completed responses only (removes preview, incomplete, header rows)
3. Extracts only essential data:
   - Q1-Q10: Rating scales (liking, beauty, time taken)
   - Q11-Q20: Pairwise A/B choices (beauty choice, liking choice, time taken)
4. Saves a clean CSV file with 81 columns

Usage:
    python transform_qualtrics.py

Input:  RIT-Human-Aesthetic-Judgment-Study_November-27-2025_14.58.csv
Output: aesthetic_judgment_cleaned.csv
"""

import pandas as pd
import numpy as np


def main():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    RAW_FILE = "../../Data/RIT-Human-Aesthetic-Judgment-Study_November-27-2025_14.58.csv"
    OUTPUT_FILE = "aesthetic_judgment_cleaned.csv"
    
    print("\n" + "="*80)
    print("AESTHETIC JUDGMENT STUDY - DATA TRANSFORMATION")
    print("="*80)
    
    # =========================================================================
    # STEP 1: LOAD RAW DATA
    # =========================================================================
    print(f"\nLoading: {RAW_FILE}")
    df_raw = pd.read_csv(RAW_FILE)
    print(f"  Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    
    # =========================================================================
    # STEP 2: FILTER FOR COMPLETED RESPONSES
    # =========================================================================
    print("\nFiltering for completed responses...")
    print(f"  Before: {df_raw.shape[0]} rows")
    
    # Keep only rows where:
    #   - Status = "IP Address" (real response, not metadata)
    #   - Finished = "TRUE" (completed survey)
    df_filtered = df_raw[
        (df_raw["Status"] == "IP Address") & 
        (df_raw["Finished"] == "TRUE")
    ].copy()
    
    print(f"  After:  {df_filtered.shape[0]} rows")
    print(f"  Removed: {df_raw.shape[0] - df_filtered.shape[0]} rows (preview/incomplete/metadata)")
    
    # =========================================================================
    # STEP 3: EXTRACT ESSENTIAL DATA
    # =========================================================================
    print("\nExtracting essential data...")
    records = []
    
    for idx, row in df_filtered.iterrows():
        record = {}
        
        # Respondent ID
        record["ResponseId"] = row["ResponseId"]
        
        # ===== RATING QUESTIONS (Q1-Q10) =====
        # Each question has:
        #   - Qk_1: liking (1-10 scale)
        #   - Qk_2: beauty (1-10 scale)
        #   - Qk_Time_Page_Submit: time taken (seconds)
        
        for q_num in range(1, 11):
            liking_col = f"Q{q_num}_1"
            beauty_col = f"Q{q_num}_2"
            time_col = f"Q{q_num}_Time_Page Submit"
            
            # Determine category (abstract or representational)
            category = "abstract" if q_num <= 5 else "representational"
            
            # Extract and convert values
            record[f"Q{q_num}_category"] = category
            record[f"Q{q_num}_liking"] = pd.to_numeric(
                row.get(liking_col), errors="coerce"
            )
            record[f"Q{q_num}_beauty"] = pd.to_numeric(
                row.get(beauty_col), errors="coerce"
            )
            record[f"Q{q_num}_time_seconds"] = pd.to_numeric(
                row.get(time_col), errors="coerce"
            )
        
        # ===== PAIRWISE COMPARISON QUESTIONS (Q11-Q20) =====
        # Each question has:
        #   - Qk_1: beauty choice (A or B)
        #   - Qk_2: liking choice (A or B)
        #   - Qk_Time_Page_Submit: time taken (seconds)
        
        for q_num in range(11, 21):
            beauty_choice_col = f"Q{q_num}_1"
            liking_choice_col = f"Q{q_num}_2"
            time_col = f"Q{q_num}_Time_Page Submit"
            
            # Determine category (abstract or representational)
            category = "abstract" if q_num <= 15 else "representational"
            
            # Extract values (keep A/B as strings)
            raw_beauty = row.get(beauty_choice_col, "")
            raw_liking = row.get(liking_choice_col, "")
            
            # Clean up: extract just 'A' or 'B' from the response
            beauty_choice = extract_choice(raw_beauty)
            liking_choice = extract_choice(raw_liking)
            
            record[f"Q{q_num}_category"] = category
            record[f"Q{q_num}_beauty_choice"] = beauty_choice
            record[f"Q{q_num}_liking_choice"] = liking_choice
            record[f"Q{q_num}_time_seconds"] = pd.to_numeric(
                row.get(time_col), errors="coerce"
            )
        
        records.append(record)
    
    # =========================================================================
    # STEP 4: CREATE CLEAN DATAFRAME
    # =========================================================================
    print(f"  Extracted {len(records)} complete respondent records")
    df_clean = pd.DataFrame(records)
    
    # =========================================================================
    # STEP 5: VALIDATE DATA
    # =========================================================================
    print("\nValidating data...")
    
    # Check rating values are in range 1-10
    rating_issues = 0
    for q in range(1, 11):
        vals = df_clean[f"Q{q}_liking"].dropna()
        if len(vals) > 0 and ((vals < 1).any() or (vals > 10).any()):
            print(f"  WARNING: Q{q}_liking has values outside [1, 10]")
            rating_issues += 1
        
        vals = df_clean[f"Q{q}_beauty"].dropna()
        if len(vals) > 0 and ((vals < 1).any() or (vals > 10).any()):
            print(f"  WARNING: Q{q}_beauty has values outside [1, 10]")
            rating_issues += 1
    
    # Check pairwise choices are A or B
    choice_issues = 0
    for q in range(11, 21):
        vals = df_clean[f"Q{q}_beauty_choice"].dropna()
        invalid = ~vals.isin(['A', 'B'])
        if invalid.any():
            print(f"  WARNING: Q{q}_beauty_choice has invalid values")
            choice_issues += 1
        
        vals = df_clean[f"Q{q}_liking_choice"].dropna()
        invalid = ~vals.isin(['A', 'B'])
        if invalid.any():
            print(f"  WARNING: Q{q}_liking_choice has invalid values")
            choice_issues += 1
    
    if rating_issues == 0 and choice_issues == 0:
        print("  ✓ All data values valid")
    
    # =========================================================================
    # STEP 6: SAVE CLEAN CSV
    # =========================================================================
    print(f"\nSaving to: {OUTPUT_FILE}")
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✓ Saved {OUTPUT_FILE}")
    print(f"  Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    
    # =========================================================================
    # STEP 7: SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "="*80)
    print("TRANSFORMATION SUMMARY")
    print("="*80)
    
    print(f"\nRATING QUESTIONS (Q1-Q10):")
    abstract_like = df_clean[[f"Q{i}_liking" for i in range(1, 6)]].values.flatten()
    abstract_like = abstract_like[~np.isnan(abstract_like)]
    rep_like = df_clean[[f"Q{i}_liking" for i in range(6, 11)]].values.flatten()
    rep_like = rep_like[~np.isnan(rep_like)]
    
    print(f"  Abstract (Q1-Q5) Liking:      Mean={np.mean(abstract_like):.2f}, SD={np.std(abstract_like):.2f}")
    
    abstract_beauty = df_clean[[f"Q{i}_beauty" for i in range(1, 6)]].values.flatten()
    abstract_beauty = abstract_beauty[~np.isnan(abstract_beauty)]
    print(f"  Abstract (Q1-Q5) Beauty:      Mean={np.mean(abstract_beauty):.2f}, SD={np.std(abstract_beauty):.2f}")
    
    print(f"  Representational (Q6-Q10) Liking:  Mean={np.mean(rep_like):.2f}, SD={np.std(rep_like):.2f}")
    
    rep_beauty = df_clean[[f"Q{i}_beauty" for i in range(6, 11)]].values.flatten()
    rep_beauty = rep_beauty[~np.isnan(rep_beauty)]
    print(f"  Representational (Q6-Q10) Beauty:  Mean={np.mean(rep_beauty):.2f}, SD={np.std(rep_beauty):.2f}")
    
    print(f"\nPAIRWISE QUESTIONS (Q11-Q20):")
    abs_beauty_b = sum(df_clean[[f"Q{i}_beauty_choice" for i in range(11, 16)]].stack() == 'B')
    abs_beauty_total = df_clean[[f"Q{i}_beauty_choice" for i in range(11, 16)]].stack().notna().sum()
    print(f"  Abstract (Q11-Q15) Beauty: B chosen {abs_beauty_b}/{abs_beauty_total} ({100*abs_beauty_b/abs_beauty_total:.1f}%)")
    
    rep_beauty_b = sum(df_clean[[f"Q{i}_beauty_choice" for i in range(16, 21)]].stack() == 'B')
    rep_beauty_total = df_clean[[f"Q{i}_beauty_choice" for i in range(16, 21)]].stack().notna().sum()
    print(f"  Representational (Q16-Q20) Beauty: B chosen {rep_beauty_b}/{rep_beauty_total} ({100*rep_beauty_b/rep_beauty_total:.1f}%)")
    
    print("\n" + "="*80)
    print("TRANSFORMATION COMPLETE!")
    print("="*80)
    print(f"\nYour cleaned data is ready in: {OUTPUT_FILE}")
    print("You can now use it for analysis, visualization, and publication.\n")


def extract_choice(value):
    """Extract 'A' or 'B' from a cell value (handles formatting variations)."""
    if pd.isna(value):
        return None
    
    value_str = str(value).upper().strip()
    
    # Try to find 'A' or 'B'
    if 'A' in value_str:
        # Check if 'A' comes before 'B' (usually it's 'A\n...B...')
        if 'B' in value_str and value_str.index('A') > value_str.index('B'):
            return 'B'
        return 'A'
    elif 'B' in value_str:
        return 'B'
    
    return None


if __name__ == "__main__":
    main()
