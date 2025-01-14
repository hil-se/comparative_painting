import random

import pandas as pd
import tensorflow as tf

pd.set_option('display.max_columns', None)

height = 250
width = 250


def loadData(col="Average", num_img=5500):
    if col == "Average":
        data = pd.read_csv("/Users/manojreddy/Documents/Comparable/Data/ImageExp/Selected_Ratings3.csv")
        data = data[["Filename", col]]
    else:
        data = pd.read_csv("/Users/manojreddy/Documents/Comparable/Data/ImageExp/All_Ratings3.csv")
        data = data[data["Rater"] == int(col)][["Filename", "Rating"]].rename(
            columns={"Filename": "Filename", "Rating": col})
    data = data.sample(frac=1)
    data = data.head(num_img)

    return data


def retrievePixels(path):
    # img = tf.keras.utils.load_img("../data/images/"+path, grayscale=False)
    folder_path = "/Users/manojreddy/Documents/Comparable/Data/Images/"
    # folder_path = "../../../XAI_Image/data/images/"
    img = tf.keras.utils.load_img(folder_path + path, target_size=(height, width))
    x = tf.keras.utils.img_to_array(img)
    return x


def processData(h=250, w=250, col="Average", num_comp=1, num_img=5500):
    height = h
    width = w
    data = loadData(col=col, num_img=num_img)
    # threshold = data["Average"].describe()["std"]
    # threshold = round(threshold.item(), 3)
    train = data.sample(frac=0.8)
    test = data.drop(train.index)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    protected_tr_race = []
    protected_ts_race = []

    protected_tr_sex = []
    protected_ts_sex = []

    # for file in train["Filename"]:
    #     if file[1] == 'M':
    #         protected_tr_sex.append(1)
    #     else:
    #         protected_tr_sex.append(0)
    #
    for file in test["Filename"]:
        if file[1] == 'M':
            protected_ts_sex.append(1)
        else:
            protected_ts_sex.append(0)
    #
    # for file in train["Filename"]:
    #     if file[0] == 'C':
    #         protected_tr_race.append(1)
    #     else:
    #         protected_tr_race.append(0)
    #
    for file in test["Filename"]:
        if file[0] == 'C':
            protected_ts_race.append(1)
        else:
            protected_ts_race.append(0)

    res_tr = []
    res_ts = []

    res_tr_single = []
    res_ts_single = []
    print("\nGenerating training data...")

    for indexA, rowA in train.iterrows():
        comp = []
        while len(comp) < num_comp:
            indexB = random.randint(0, len(train) - 1)
            rowB = train.iloc[indexB]
            if (indexA == indexB) or (indexB in comp):
                continue
            ratingA = rowA[col]
            ratingB = rowB[col]
            label = 0
            if ratingA > ratingB:
                label = 1
            elif ratingA < ratingB:
                label = -1
            if label != 0:
                res_tr.append({"A": rowA["Filename"],
                               "B": rowB["Filename"],
                               "Label": label
                               })

                res_tr_single.append({"A": rowA["Filename"],
                                      "B": rowB["Filename"],
                                      "Label": label
                                      })

                res_tr.append({"A": rowB["Filename"],
                               "B": rowA["Filename"],
                               "Label": -label
                               })
                comp.append(indexB)
    data_tr = pd.DataFrame(res_tr)
    data_tr_single = pd.DataFrame(res_tr_single)

    data_tr['A'] = data_tr['A'].apply(retrievePixels).div(255.0)
    data_tr['B'] = data_tr['B'].apply(retrievePixels).div(255.0)

    data_tr_single['A'] = data_tr_single['A'].apply(retrievePixels).div(255.0)
    data_tr_single['B'] = data_tr_single['B'].apply(retrievePixels).div(255.0)
    # print("Saving training data...")
    # data_tr = data_tr.sample(frac=1)
    # data_tr.to_csv("../../Data/ImageExp/image_train.csv", index=False)
    print("Generating testing data...")
    for indexA, rowA in test.iterrows():
        comp = []
        # for indexB, rowB in test.iterrows():
        #     if (indexA == indexB) or (protected_ts[indexA] == protected_ts[indexB]):
        #         continue
        while len(comp) < num_comp:
            indexB = random.randint(0, len(test) - 1)
            rowB = test.iloc[indexB]
            if (indexA == indexB) or (indexB in comp):
                continue
            ratingA = rowA[col]
            ratingB = rowB[col]
            label = 0
            if ratingA > ratingB:
                label = 1
            elif ratingA < ratingB:
                label = -1
            if label != 0:
                res_ts.append({"A": rowA["Filename"],
                               "B": rowB["Filename"],
                               "Label": label
                               })
                res_ts_single.append({"A": rowA["Filename"],
                                      "B": rowB["Filename"],
                                      "Label": label
                                      })
                res_ts.append({"A": rowB["Filename"],
                               "B": rowA["Filename"],
                               "Label": -label
                               })
                comp.append(indexB)
    data_ts = pd.DataFrame(res_ts)
    data_ts_single = pd.DataFrame(res_ts_single)

    protected_ts_A_sex = []
    protected_ts_B_sex = []

    protected_ts_A_race = []
    protected_ts_B_race = []

    protected_ts_A_sex_single = []
    protected_ts_B_sex_single = []

    protected_ts_A_race_single = []
    protected_ts_B_race_single = []

    for file in data_ts["A"]:
        if file[1] == 'M':
            protected_ts_A_sex.append(1)
        else:
            protected_ts_A_sex.append(0)

    for file in data_ts["B"]:
        if file[1] == 'M':
            protected_ts_B_sex.append(1)
        else:
            protected_ts_B_sex.append(0)

    for file in data_ts["A"]:
        if file[0] == 'C':
            protected_ts_A_race.append(1)
        else:
            protected_ts_A_race.append(0)

    for file in data_ts["B"]:
        if file[0] == 'C':
            protected_ts_B_race.append(1)
        else:
            protected_ts_B_race.append(0)

    protected_ts_AB_race = pd.DataFrame({
        "A": protected_ts_A_race,
        "B": protected_ts_B_race
    })

    protected_ts_AB_sex = pd.DataFrame({
        "A": protected_ts_A_sex,
        "B": protected_ts_B_sex
    })

    for file in data_ts_single["A"]:
        if file[1] == 'M':
            protected_ts_A_sex_single.append(1)
        else:
            protected_ts_A_sex_single.append(0)

    for file in data_ts_single["B"]:
        if file[1] == 'M':
            protected_ts_B_sex_single.append(1)
        else:
            protected_ts_B_sex_single.append(0)

    for file in data_ts_single["A"]:
        if file[0] == 'C':
            protected_ts_A_race_single.append(1)
        else:
            protected_ts_A_race_single.append(0)

    for file in data_ts_single["B"]:
        if file[0] == 'C':
            protected_ts_B_race_single.append(1)
        else:
            protected_ts_B_race_single.append(0)

    protected_ts_AB_race_single = pd.DataFrame({
        "A": protected_ts_A_race_single,
        "B": protected_ts_B_race_single
    })

    protected_ts_AB_sex_single = pd.DataFrame({
        "A": protected_ts_A_sex_single,
        "B": protected_ts_B_sex_single
    })

    data_ts['A'] = data_ts['A'].apply(retrievePixels).div(255.0)
    data_ts['B'] = data_ts['B'].apply(retrievePixels).div(255.0)

    data_ts_single['A'] = data_ts_single['A'].apply(retrievePixels).div(255.0)
    data_ts_single['B'] = data_ts_single['B'].apply(retrievePixels).div(255.0)

    # print("Saving testing data...")
    # data_ts = data_ts.sample(frac=1)
    # data_ts.to_csv("../../Data/ImageExp/image_test.csv", index=False)
    print("Done.")
    print("Training data size:", len(data_tr.index))
    print("Testing data size:", len(data_ts.index))
    print("Training data size:", len(data_tr_single.index))
    print("Testing data size:", len(data_ts_single.index))

    test_list = []
    for indexA, rowA in test.iterrows():
        ratingA = rowA[col]
        test_list.append({"indexA": indexA, "A": rowA["Filename"], "Score": ratingA})
    test_list = pd.DataFrame(test_list)
    test_list['A'] = test_list['A'].apply(retrievePixels)

    data_list = []
    for indexA, rowA in data.iterrows():
        ratingA = rowA[col]
        data_list.append({"indexA": indexA, "A": rowA["Filename"], "Score": ratingA})
    data_list = pd.DataFrame(data_list)
    data_list['A'] = data_list['A'].apply(retrievePixels)

    return (data_tr, data_ts, data_tr_single, data_ts_single, test_list, data_list, len(data_tr.index), len(
        data_ts.index), len(data_tr_single.index), len(
        data_ts_single.index), train, test, protected_ts_race, protected_ts_sex, protected_ts_AB_race,
            protected_ts_AB_sex, protected_ts_AB_race_single, protected_ts_AB_sex_single)
