import pandas as pd
import numpy as np
import os



# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_PATH = "../../Data/RIT-Human-Aesthetic-Judgment-Study_November-27-2025_14.58.csv"
OUTPUT_DIR = "../../results/human_survey/survey_data"



# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv(CSV_PATH)

# Keep only TRUE finished, non-preview responses
df = df[(df["Finished"] == "TRUE") & (df["Status"] != "Survey Preview")].copy()

print("="*80)
print("TRANSFORMING SURVEY DATA")
print("="*80)
print(f"\nCompleted responses (non-preview): {len(df)}\n")

# Create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)



# ============================================================================
# GROUND TRUTH VALUES - ABSOLUTE
# ============================================================================

# First 5 = abstract (Q1–Q5), next 5 = representational (Q6–Q10)
gt_beauty_values = [
    4.42, 3.67, 2.79, 7.00, 6.69,   # Abstract Beauty (Q1–Q5)
    4.84, 2.80, 6.71, 5.88, 4.02    # Repr Beauty (Q6–Q10)
]

gt_liking_values = [
    4.45, 4.18, 4.30, 6.95, 4.22,   # Abstract Liking (Q1–Q5)
    4.80, 3.10, 6.29, 5.61, 3.70    # Repr Liking (Q6–Q10)
]



# ============================================================================
# GROUND TRUTH VALUES - COMPARATIVE
# ============================================================================

# For each question 11–20: (A_beauty, A_liking), (B_beauty, B_liking)
gt_comp_pairs = [
    # Q11
    (4.25, 4.68), (3.63, 4.43),
    # Q12
    (3.49, 4.37), (4.02, 6.17),
    # Q13
    (4.80, 4.73), (3.21, 4.15),
    # Q14
    (3.77, 4.95), (3.04, 4.31),
    # Q15
    (3.91, 4.31), (4.85, 5.68),

    # Q16
    (4.80, 4.59), (4.53, 3.95),
    # Q17
    (6.00, 5.57), (5.67, 5.27),
    # Q18
    (6.31, 6.68), (4.00, 4.17),
    # Q19
    (6.63, 6.68), (5.84, 6.12),
    # Q20
    (3.59, 3.42), (4.78, 4.28),
]

# Split into A and B arrays, then into beauty/liking
gt_A = gt_comp_pairs[0::2]   # (A_beauty, A_liking) for Q11–Q20
gt_B = gt_comp_pairs[1::2]   # (B_beauty, B_liking) for Q11–Q20

gt_A_beauty = [x[0] for x in gt_A]
gt_A_liking = [x[1] for x in gt_A]
gt_B_beauty = [x[0] for x in gt_B]
gt_B_liking = [x[1] for x in gt_B]

# Helper slices: first 5 = abstract (Q11–Q15), last 5 = repr (Q16–Q20)
A_beauty_abs = gt_A_beauty[:5]
A_liking_abs = gt_A_liking[:5]
B_beauty_abs = gt_B_beauty[:5]
B_liking_abs = gt_B_liking[:5]

A_beauty_repr = gt_A_beauty[5:]
A_liking_repr = gt_A_liking[5:]
B_beauty_repr = gt_B_beauty[5:]
B_liking_repr = gt_B_liking[5:]



# ============================================================================
# ABSOLUTE RATINGS - 4 FILES
# ============================================================================

print("Creating Absolute Rating CSVs...")


# 1. Abstract Beauty (Q1–Q5, column Q?_2)
print("  1. Abstract_Beauty_Absolute")
abstract_beauty_cols = [f"Q{i}_2" for i in range(1, 6)]
abs_beauty_data = []

for idx, row in df.iterrows():
    vals = pd.to_numeric(row[abstract_beauty_cols], errors="coerce").values
    abs_beauty_data.append(vals)

abs_beauty_df = pd.DataFrame(abs_beauty_data).T
abs_beauty_df.columns = [f"P{i+1}" for i in range(len(abs_beauty_data))]
abs_beauty_df.insert(0, "Task", [f"Q{i}" for i in range(1, 6)])

# GT column for abstract beauty (first 5 beauty values)
abs_beauty_df["GT"] = gt_beauty_values[:5]

abs_beauty_df["AVG_P_Rating"] = abs_beauty_df[
    [c for c in abs_beauty_df.columns if c.startswith("P")]
].mean(axis=1)
abs_beauty_df.to_csv(os.path.join(OUTPUT_DIR, "Abstract_Beauty_Absolute.csv"), index=False)


# 2. Abstract Liking (Q1–Q5, column Q?_1)
print("  2. Abstract_Liking_Absolute")
abstract_liking_cols = [f"Q{i}_1" for i in range(1, 6)]
abs_liking_data = []

for idx, row in df.iterrows():
    vals = pd.to_numeric(row[abstract_liking_cols], errors="coerce").values
    abs_liking_data.append(vals)

abs_liking_df = pd.DataFrame(abs_liking_data).T
abs_liking_df.columns = [f"P{i+1}" for i in range(len(abs_liking_data))]
abs_liking_df.insert(0, "Task", [f"Q{i}" for i in range(1, 6)])

# GT column for abstract liking (first 5 liking values)
abs_liking_df["GT"] = gt_liking_values[:5]

abs_liking_df["AVG_P_Rating"] = abs_liking_df[
    [c for c in abs_liking_df.columns if c.startswith("P")]
].mean(axis=1)
abs_liking_df.to_csv(os.path.join(OUTPUT_DIR, "Abstract_Liking_Absolute.csv"), index=False)


# 3. Representational Beauty (Q6–Q10, column Q?_2)
print("  3. Repr_Beauty_Absolute")
repr_beauty_cols = [f"Q{i}_2" for i in range(6, 11)]
repr_beauty_data = []

for idx, row in df.iterrows():
    vals = pd.to_numeric(row[repr_beauty_cols], errors="coerce").values
    repr_beauty_data.append(vals)

repr_beauty_df = pd.DataFrame(repr_beauty_data).T
repr_beauty_df.columns = [f"P{i+1}" for i in range(len(repr_beauty_data))]
repr_beauty_df.insert(0, "Task", [f"Q{i}" for i in range(6, 11)])

# GT column for repr beauty (last 5 beauty values)
repr_beauty_df["GT"] = gt_beauty_values[5:]

repr_beauty_df["AVG_P_Rating"] = repr_beauty_df[
    [c for c in repr_beauty_df.columns if c.startswith("P")]
].mean(axis=1)
repr_beauty_df.to_csv(os.path.join(OUTPUT_DIR, "Repr_Beauty_Absolute.csv"), index=False)


# 4. Representational Liking (Q6–Q10, column Q?_1)
print("  4. Repr_Liking_Absolute")
repr_liking_cols = [f"Q{i}_1" for i in range(6, 11)]
repr_liking_data = []

for idx, row in df.iterrows():
    vals = pd.to_numeric(row[repr_liking_cols], errors="coerce").values
    repr_liking_data.append(vals)

repr_liking_df = pd.DataFrame(repr_liking_data).T
repr_liking_df.columns = [f"P{i+1}" for i in range(len(repr_liking_data))]
repr_liking_df.insert(0, "Task", [f"Q{i}" for i in range(6, 11)])

# GT column for repr liking (last 5 liking values)
repr_liking_df["GT"] = gt_liking_values[5:]

repr_liking_df["AVG_P_Rating"] = repr_liking_df[
    [c for c in repr_liking_df.columns if c.startswith("P")]
].mean(axis=1)
repr_liking_df.to_csv(os.path.join(OUTPUT_DIR, "Repr_Liking_Absolute.csv"), index=False)



# ============================================================================
# COMPARATIVE JUDGMENTS - 4 FILES (WITH GT_A / GT_B)
# ============================================================================

print("\nCreating Comparative Judgment CSVs...")


# 5. Abstract Beauty (Q11–Q15, column Q?_1)
print("  5. Abstract_Beauty_Comparative")
comp_abs_beauty_cols = [f"Q{i}_1" for i in range(11, 16)]
comp_abs_beauty_data = []

for idx, row in df.iterrows():
    vals = row[comp_abs_beauty_cols].values
    comp_abs_beauty_data.append(vals)

comp_abs_beauty_df = pd.DataFrame(comp_abs_beauty_data).T
comp_abs_beauty_df.columns = [f"P{i+1}" for i in range(len(comp_abs_beauty_data))]
comp_abs_beauty_df.insert(0, "Task", [f"Q{i}" for i in range(11, 16)])

# GT_A / GT_B for abstract beauty (Q11–Q15)
comp_abs_beauty_df["GT_A"] = A_beauty_abs
comp_abs_beauty_df["GT_B"] = B_beauty_abs

comp_abs_beauty_df.to_csv(os.path.join(OUTPUT_DIR, "Abstract_Beauty_Comparative.csv"), index=False)


# 6. Abstract Liking (Q11–Q15, column Q?_2)
print("  6. Abstract_Liking_Comparative")
comp_abs_liking_cols = [f"Q{i}_2" for i in range(11, 16)]
comp_abs_liking_data = []

for idx, row in df.iterrows():
    vals = row[comp_abs_liking_cols].values
    comp_abs_liking_data.append(vals)

comp_abs_liking_df = pd.DataFrame(comp_abs_liking_data).T
comp_abs_liking_df.columns = [f"P{i+1}" for i in range(len(comp_abs_liking_data))]
comp_abs_liking_df.insert(0, "Task", [f"Q{i}" for i in range(11, 16)])

# GT_A / GT_B for abstract liking (Q11–Q15)
comp_abs_liking_df["GT_A"] = A_liking_abs
comp_abs_liking_df["GT_B"] = B_liking_abs

comp_abs_liking_df.to_csv(os.path.join(OUTPUT_DIR, "Abstract_Liking_Comparative.csv"), index=False)


# 7. Representational Beauty (Q16–Q20, column Q?_1)
print("  7. Repr_Beauty_Comparative")
comp_repr_beauty_cols = [f"Q{i}_1" for i in range(16, 21)]
comp_repr_beauty_data = []

for idx, row in df.iterrows():
    vals = row[comp_repr_beauty_cols].values
    comp_repr_beauty_data.append(vals)

comp_repr_beauty_df = pd.DataFrame(comp_repr_beauty_data).T
comp_repr_beauty_df.columns = [f"P{i+1}" for i in range(len(comp_repr_beauty_data))]
comp_repr_beauty_df.insert(0, "Task", [f"Q{i}" for i in range(16, 21)])

# GT_A / GT_B for repr beauty (Q16–Q20)
comp_repr_beauty_df["GT_A_Beauty"] = A_beauty_repr
comp_repr_beauty_df["GT_B_Beauty"] = B_beauty_repr

comp_repr_beauty_df.to_csv(os.path.join(OUTPUT_DIR, "Repr_Beauty_Comparative.csv"), index=False)


# 8. Representational Liking (Q16–Q20, column Q?_2)
print("  8. Repr_Liking_Comparative")
comp_repr_liking_cols = [f"Q{i}_2" for i in range(16, 21)]
comp_repr_liking_data = []

for idx, row in df.iterrows():
    vals = row[comp_repr_liking_cols].values
    comp_repr_liking_data.append(vals)

comp_repr_liking_df = pd.DataFrame(comp_repr_liking_data).T
comp_repr_liking_df.columns = [f"P{i+1}" for i in range(len(comp_repr_liking_data))]
comp_repr_liking_df.insert(0, "Task", [f"Q{i}" for i in range(16, 21)])

# GT_A / GT_B for repr liking (Q16–Q20)
comp_repr_liking_df["GT_A"] = A_liking_repr
comp_repr_liking_df["GT_B"] = B_liking_repr

comp_repr_liking_df.to_csv(os.path.join(OUTPUT_DIR, "Repr_Liking_Comparative.csv"), index=False)



# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

summary_data = [
    ["Abstract_Beauty_Absolute", len(abs_beauty_data), "5 questions (Q1–Q5)"],
    ["Abstract_Liking_Absolute", len(abs_liking_data), "5 questions (Q1–Q5)"],
    ["Repr_Beauty_Absolute", len(repr_beauty_data), "5 questions (Q6–Q10)"],
    ["Repr_Liking_Absolute", len(repr_liking_data), "5 questions (Q6–Q10)"],
    ["Abstract_Beauty_Comparative", len(comp_abs_beauty_data), "5 questions (Q11–Q15)"],
    ["Abstract_Liking_Comparative", len(comp_abs_liking_data), "5 questions (Q11–Q15)"],
    ["Repr_Beauty_Comparative", len(comp_repr_beauty_data), "5 questions (Q16–Q20)"],
    ["Repr_Liking_Comparative", len(comp_repr_liking_data), "5 questions (Q16–Q20)"],
]

print("\nCSV files created in folder:", OUTPUT_DIR)
for i, (sheet_name, n_participants, questions) in enumerate(summary_data, 1):
    print(f"  {i}. {sheet_name:<30} {n_participants} participants, {questions}")

print("\n" + "="*80)
print("DONE!")
print("="*80)
