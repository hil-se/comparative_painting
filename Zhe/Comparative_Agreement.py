import numpy as np
import pandas as pd
from collections import Counter
from pdb import set_trace


def generate_pairs(x1, x2):
    n = len(x1)
    pairs = {"A":[], "B":[], "agree": []}
    for i in range(n):
        for j in range(i+1, n):
            d1 = d2 = 0
            if x1[i]>x1[j]:
                d1 = 1
            elif x1[i]<x1[j]:
                d1 = -1
            if x2[i]>x2[j]:
                d2 = 1
            elif x1[i]<x1[j]:
                d2 = -1
            if d1!=0 and d2!=0:
                pairs["A"].append(d1)
                pairs["B"].append(d2)
                pairs["agree"].append(d1==d2)
    return pairs

def acc_kappa(pairs):
    n = len(pairs["agree"])
    acc = np.sum(pairs["agree"]) / n
    count_A = Counter(pairs["A"])
    count_B = Counter(pairs["B"])
    pe = n**(-2)*(count_A[1]*count_B[1]+count_A[-1]*count_B[-1])
    kappa = 1-(1-acc)/(1-pe)
    return acc, kappa


df = pd.read_csv(
    "human_survey_data/Abstract_Beauty_Absolute.csv"
    # "human_survey_data/Abstract_Liking_Absolute.csv"
    # "human_survey_data/Repr_Liking_Absolute.csv"
    # "human_survey_data/Repr_Liking_Absolute.csv"
    
    )
raters = ["P1","P3","P4","P5","P6"]
results = []
for i in range(len(raters)):
    for j in range(i+1, len(raters)):
        pairs = generate_pairs(df[raters[i]], df[raters[j]])
        acc, kappa = acc_kappa(pairs)
        result = {"Pair": raters[i]+"/"+raters[j], "Acc": "%.2f" %acc, "Kappa": "%.2f" %kappa}
        results.append(result)

result_df = pd.DataFrame(results)
result_df.to_csv(
    "human_survey_data/agreements_absolute/Agreement_Abstract_Beauty_Absolute.csv", index=False
    # "human_survey_data/agreements_absolute/Agreement_Abstract_Liking_Absolute.csv", index=False
    # "human_survey_data/agreements_absolute/Agreement_Repr_Liking_Absolute.csv", index=False
    # "human_survey_data/agreements_absolute/Agreement_Repr_Liking_Absolute.csv", index=False

    )
print(result_df)