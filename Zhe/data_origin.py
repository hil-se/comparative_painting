import tensorflow as tf
import os
import numpy as np
import pandas as pd
from pdb import set_trace

def load_data(path = "../Data/Abstract_Images/"):
    features = []
    for id in range(240):
        name = id+1
        if name < 10:
            name = "0"+str(name)
        else:
            name = str(name)
        if os.path.exists(path+name+".jpg"):
            img = tf.keras.utils.load_img(path+name+".jpg", color_mode='rgb')
        elif os.path.exists(path+name+"cropped.jpg"):
            img = tf.keras.utils.load_img(path + name + "cropped.jpg", color_mode='rgb')
        elif os.path.exists(path+name+"croppedtofit.jpg"):
            img = tf.keras.utils.load_img(path + name + "croppedtofit.jpg", color_mode='rgb')
        elif os.path.exists(path+name+".jpeg"):
            img = tf.keras.utils.load_img(path + name + ".jpeg", color_mode='rgb')
        elif os.path.exists(path+name+"cropped.jpeg"):
            img = tf.keras.utils.load_img(path + name + "cropped.jpeg", color_mode='rgb')
        else:
            print(id)
            continue
        img_array = tf.keras.utils.img_to_array(img)
        model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                                                        input_shape=img_array.shape, pooling="avg")
        feature = model(tf.Variable([img_array])).numpy()[0]
        features.append(list(feature))
    return np.array(features)

def rating():
    id_abs = list(set(range(1,241)) - set([173]))
    id_rep = list(set(range(1,241)) - set([90,157]))

    df = pd.read_csv("../Data/Abstract_All_Raters.csv")
    data = {"Painting": []}
    for i, id in enumerate(id_abs):
        tmp = df[df["Painting"]==str(id)+".jpg"]
        data["Painting"].append(i)
        for j in range(len(tmp)):
            rater = tmp["Rater"].iloc[j]
            if rater not in data:
                data[rater]=[]
            data[rater].append(tmp["Beauty"].iloc[j])
    pd.DataFrame(data).to_csv("feature/abstract_beauty.csv",index=False)

    df = pd.read_csv("../Data/Abstract_Liking_All_Raters.csv")
    data = {"Painting": []}
    for i, id in enumerate(id_abs):
        tmp = df[df["Painting"] == str(id) + ".jpg"]
        data["Painting"].append(i)
        for j in range(len(tmp)):
            rater = tmp["Rater"].iloc[j]
            if rater not in data:
                data[rater] = []
            data[rater].append(tmp["Liking"].iloc[j])
    pd.DataFrame(data).to_csv("feature/abstract_liking.csv", index=False)

    df = pd.read_csv("../Data/Representational_All_Raters.csv")
    data = {"Painting": []}
    for i, id in enumerate(id_rep):
        tmp = df[df["Painting"] == str(id) + ".jpg"]
        data["Painting"].append(i)
        for j in range(len(tmp)):
            rater = tmp["Rater"].iloc[j]
            if rater not in data:
                data[rater] = []
            data[rater].append(tmp["Beauty"].iloc[j])
    pd.DataFrame(data).to_csv("feature/representational_beauty.csv", index=False)

    df = pd.read_csv("../Data/Representational_Liking_All_Raters.csv")
    data = {"Painting": []}
    for i, id in enumerate(id_rep):
        tmp = df[df["Painting"] == str(id) + ".jpg"]
        data["Painting"].append(i)
        for j in range(len(tmp)):
            rater = tmp["Rater"].iloc[j]
            if rater not in data:
                data[rater] = []
            data[rater].append(tmp["Liking"].iloc[j])
    pd.DataFrame(data).to_csv("feature/representational_liking.csv", index=False)


if __name__ == '__main__':
    path = "../Data/Abstract_Images/"
    features = load_data(path)
    np.save('feature/abstract_feature_origin.npy', features)
    path = "../Data/Representational_Images/"
    features = load_data(path)
    np.save('feature/representational_feature_origin.npy', features)

    rating()



