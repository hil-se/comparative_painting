import pandas as pd

def generate_pairs(df):
    res_tr = []
    for indexA in range(0,len(df)):
        comp = []
        rowA = df.iloc[indexA]
        # while len(comp) < num_comp:
        #     indexB = random.randint(0, len(df) - 1)
        #     rowB = df.iloc[indexB]
        for indexB in range(0 , len(df)):
            # if (indexA == indexB) or (indexB in comp):
            #     continue
            rowB = df.iloc[indexB]
            ratingA = rowA["Y"]
            ratingB = rowB["Y"]
            predA = rowA["Y_pred"]
            predB = rowB["Y_pred"]
            label = 0
            pred = 0
            if ratingA > ratingB:
                label = 1
            elif ratingA < ratingB:
                label = -1
            if predA > predB:
                pred = 1
            elif predA < predB:
                pred = -1
            res_tr.append({"A": rowA["A"],
                           "B": rowB["A"],
                           "Label": label,
                           "pred": pred
                           })
            comp.append(indexB)
    data_tr = pd.DataFrame(res_tr)
    return data_tr