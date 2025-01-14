import os
import random

import pandas as pd
from sklearn.svm import LinearSVC

import Classification

path = "../../Data/landmark_txt/"
os.chdir(path)

import metrics

final_results = []
final_results_encoder = []

dataName = "SCUT"
iterations = 5
num_comp = 1
global col

for i in range(iterations):
    print("Iteration:", i + 1, "/", iterations)

    if dataName == "Boston":
        df = pd.read_csv("../../Data/Boston.csv")
        col = "MEDV"
        features = list(df.columns)
        for feature in features:
            if feature != col:
                df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

    elif dataName == "WorldHappiness":
        df = pd.read_csv("../../Data/WorldHappiness.csv")
        col = 'WorldHappiness2022'
        df["unMember"] = df["unMember"].apply(lambda x: int(x))
        regions = df["region"].unique()
        for region in regions:
            df[region] = 0
        for index, row in df.iterrows():
            region = str(row["region"])
            df.loc[index, region] = 1
        df = df.drop(["share_borders", "region", "country"], axis=1)
        features = list(df.columns)
        for feature in features:
            if feature == "WorldHappiness2022" or feature == "country":
                continue
            df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
        df.rename(
            columns={"population_2024": "Population", "population_growthRate": "Growth rate", "land_area": "Land area",
                     "unMember": "UN member", "population_density": "Density",
                     "Hdi2021": "HDI21", "Hdi2020": "HDI20"}, inplace=True)

    elif dataName == "SCUT":

        df = []
        col = 'Average'

        All_labels = pd.read_csv("../../Data/ImageExp/Selected_Ratings.csv")

        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".txt"):
                lm = pd.read_csv(path + file, sep=" ", header=None).to_numpy().flatten().tolist()
                label = All_labels.loc[All_labels['Filename'] == file.replace(".txt", '.jpg')][col].values[0]
                lm.append(label)
                df.append(lm)

        df = pd.DataFrame(df)
        df.rename(columns={172: col}, inplace=True)
        features = list(df.columns)
        for feature in features:
            if feature != col:
                df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

    train = df.sample(frac=0.8)
    test = df.drop(train.index)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    res_tr = []
    res_ts = []

    res_tr_encoder = []
    res_ts_encoder = []
    test_list = []
    protected_ts = []

    for indexA, rowA in train.iterrows():
        comp = []
        while len(comp) < num_comp:
            indexB = random.randint(0, len(train) - 1)
            rowB = train.iloc[indexB]
            if (indexA == indexB) or (indexB in comp):
                continue
            # for indexB, rowB in train.iterrows():
            #     if (indexA == indexB):
            #         continue
            ratingA = rowA[col]
            ratingB = rowB[col]
            label = 0
            if ratingA > ratingB:
                label = 1
            elif ratingA < ratingB:
                label = -1
            if label != 0:
                trainA = rowA.drop(labels=[col])
                trainB = rowB.drop(labels=[col])
                res_tr.append(pd.concat([trainA - trainB, pd.Series(label, index=["label"])]))
                res_tr.append(pd.concat([trainB - trainA, pd.Series(-label, index=["label"])]))

                res_tr_encoder.append({"A": trainA.to_list(),
                                       "B": trainB.to_list(),
                                       "Label": label
                                       })
                res_tr_encoder.append({"A": trainB.to_list(),
                                       "B": trainA.to_list(),
                                       "Label": -label
                                       })

                comp.append(indexB)

    data_tr = pd.DataFrame(res_tr)
    data_tr_encoder = pd.DataFrame(res_tr_encoder)

    for indexA, rowA in test.iterrows():
        comp = []
        while len(comp) < num_comp:
            indexB = random.randint(0, len(test) - 1)
            rowB = test.iloc[indexB]
            if (indexA == indexB) or (indexB in comp):
                continue
            # for indexB, rowB in test.iterrows():
            #     if (indexA == indexB):
            #         continue
            ratingA = rowA[col]
            ratingB = rowB[col]
            label = 0
            if ratingA > ratingB:
                label = 1
            elif ratingA < ratingB:
                label = -1
            if label != 0:
                testA = rowA.drop(labels=[col])
                testB = rowB.drop(labels=[col])
                res_ts.append(pd.concat([testA - testB, pd.Series(label, index=["label"])]))
                res_ts.append(pd.concat([testB - testA, pd.Series(-label, index=["label"])]))

                res_ts_encoder.append({"A": testA.to_list(),
                                       "B": testB.to_list(),
                                       "Label": label
                                       })
                res_ts_encoder.append({"A": testB.to_list(),
                                       "B": testA.to_list(),
                                       "Label": -label
                                       })

                comp.append(indexB)

        toAdd = {"indexA": indexA, "A": rowA.drop(labels=[col]).to_list(), "Score": rowA[col]}
        test_list.append(toAdd)

    data_ts = pd.DataFrame(res_ts)
    data_ts_encoder = pd.DataFrame(res_ts_encoder)
    test_list = pd.DataFrame(test_list)

    data_tr_X = data_tr.drop(columns=["label"])
    # clf = SVC(kernel="linear")
    clf = LinearSVC()
    clf.fit(data_tr_X, data_tr["label"])

    data_ts_X = data_ts.drop(columns=["label"])
    pred = clf.predict(data_ts_X)

    test_X = test.drop(columns=col)
    pred_reg = clf.decision_function(test_X)

    m_comp = metrics.Metrics(data_ts["label"], pred)
    m_reg = metrics.Metrics(test[col], pred_reg)

    # use decision function on original dataset and caculate p_coef
    # compare with our cnn model

    result = {"Full data size": len(train), "Testing data size": len(test),
              "Recall": m_comp.recall(), "Precision": m_comp.precision(), "F1": m_comp.f1(),
              "Accuracy": m_comp.accuracy(), "Spearman coef": m_reg.spearmanr_coefficient(),
              "Spearman P": m_reg.spearmanr_value(),
              "Pearson coef": m_reg.pearsonr_coefficient(), "Pearson P": m_reg.pearsonr_value()}

    final_results.append(result)

    train_encoder = data_tr_encoder.sample(frac=0.85)
    y_true = train_encoder["Label"].tolist()
    val = data_tr_encoder.drop(train_encoder.index)

    dual_encoder = Classification.train_model(train=train_encoder, val=val, y_true=y_true, shared=True, epochs=500)

    recall, precision, F1, accuracy = Classification.test_model(data_ts_encoder, dual_encoder)
    spearmanr, sp_pvalue, pearsonr, p_pvalue = Classification.generateLists(test_list, dual_encoder)

    result_encoder = {"Full data size": len(train), "Testing data size": len(test),
                      "Recall": recall, "Precision": precision, "F1": F1,
                      "Accuracy": accuracy, "Spearman coef": spearmanr,
                      "Spearman P": sp_pvalue,
                      "Pearson coef": pearsonr, "Pearson P": p_pvalue}

    final_results_encoder.append(result_encoder)

final_results = pd.DataFrame(final_results)
final_results.to_csv("../../Results/" + dataName + " SVM_" + col + "_" + str(num_comp) + ".csv", index=False)

final_results_encoder = pd.DataFrame(final_results_encoder)
final_results_encoder.to_csv("../../Results/" + dataName + " Encoder_" + col + "_" + str(num_comp) + ".csv",
                             index=False)

# debug the encoder
# experiment on face beauty data
# change cnn to a naive linear model (train with SGD)
# include pearson's coef
# create pairs of comparison for dual encoder model
# try different batch size
# try different encoder structure % optimizer (non-linear layer)
# Redo the experiment in Table 4.1 with the same training set for different SA


# four accuracies (within & cross)
# conditional independence
