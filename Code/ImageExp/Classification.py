import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

import DataProcessing
import SharedDualEncoder
import metrics
import vgg_pre


def learn(train_data,
          epochs=100,
          validation_data=None,
          y_true=None,
          patience=10,
          batch_size=4,
          shared=False,
          height=250,
          width=250):
    if y_true is None:
        y_true = []
    td_s = train_data["A"].to_list()
    td_t = train_data["B"].to_list()
    train_y = np.array(train_data["Label"].tolist())
    source = np.array([emb for emb in td_s])
    target = np.array([emb for emb in td_t])
    tr_feature = {"A": source, "B": target, "Label": train_y}

    v_s = validation_data["A"].to_list()
    v_t = validation_data["B"].to_list()
    val_y = np.array(validation_data["Label"].tolist())
    source = np.array([emb for emb in v_s])
    target = np.array([emb for emb in v_t])
    v_feature = {"A": source, "B": target, "Label": val_y}

    train_dataset = tf.data.Dataset.from_tensor_slices(tr_feature)
    val_dataset = tf.data.Dataset.from_tensor_slices(v_feature)

    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    # strategy = tf.distribute.MirroredStrategy(
    # cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # with strategy.scope():

    encoder = SharedDualEncoder.create_encoder(height=height, width=width)
    dual_encoder = SharedDualEncoder.DualEncoderAll(encoder, y_true=np.array(y_true))
    dual_encoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    dual_encoder.fit(
        x=train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[early_stopping],
        verbose=1
    )
    return dual_encoder


def train_model(train, val, y_true, epochs=100, shared=False, height=250, width=250):
    np.random.shuffle(train.values)
    np.random.shuffle(val.values)
    dual_encoder = learn(train, epochs=epochs, validation_data=val, y_true=y_true, shared=shared, height=height,
                         width=width)
    return dual_encoder


def test_model(test, dual_encoder, protected):
    dataA = test["A"].tolist()
    dataB = test["B"].tolist()
    labels = test["Label"].tolist()
    ln = len(dataA)
    predictions = []
    for i in range(ln):
        datapoint_A = np.array(dataA[i])
        datapoint_B = np.array(dataB[i])
        datapoint_A = np.expand_dims(datapoint_A, axis=0)
        datapoint_B = np.expand_dims(datapoint_B, axis=0)
        prediction = dual_encoder.predict(datapoint_A, datapoint_B)

        # prediction = round(prediction.numpy()[0][0].item()) # Labels in [0, 1]

        prediction = prediction.numpy()[0][0].item()  # Labels in [-1, 1]
        if prediction < 0:
            prediction = -1
        else:
            prediction = 1
        # print(prediction, labels[i])

        predictions.append(prediction)
    return evaluate(labels, predictions, protected)


def evaluate(y_true, y_pred, protected):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    recall = 0
    precision = 0
    F1 = 0
    accuracy = 0
    ln = len(y_true)
    for i in range(ln):
        label = y_true[i]
        prediction = y_pred[i]
        if label == 1:
            if prediction == 1:
                TP += 1
            else:
                FN += 1
        else:
            if prediction == 1:
                FP += 1
            else:
                TN += 1
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    if (recall + precision) != 0:
        F1 = (2 * recall * precision) / (recall + precision)
    if (TP + FP + TN + FN) != 0:
        accuracy = (TP + TN) / (TP + FP + TN + FN)

    T_i_j = 0
    TP_i_j = 0
    FP_i_j = 0
    T_j_i = 0
    TP_j_i = 0
    FP_j_i = 0
    F_i_j = 0
    F_j_i = 0

    for i, row in protected.iterrows():
        if row['A'] > row['B']:
            if y_true[i] == 1:
                T_i_j += 1
                if y_pred[i] == 1:
                    TP_i_j += 1
            else:
                F_i_j += 1
                if y_pred[i] == 1:
                    FP_i_j += 1

        elif row['A'] < row['B']:
            if y_true[i] != 1:
                T_j_i += 1
                if y_pred[i] != 1:
                    TP_j_i += 1
            else:
                F_j_i += 1
                if y_pred[i] != 1:
                    FP_j_i += 1

    TPR_i_j = TP_i_j / T_i_j
    FPR_i_j = FP_i_j / F_i_j
    TPR_j_i = TP_j_i / T_j_i
    FPR_j_i = FP_j_i / F_j_i

    AOD = (TPR_i_j + FPR_i_j - TPR_j_i - FPR_j_i) / 2

    return recall, precision, F1, accuracy, AOD


def generateLists(test_dataset, dual_encoder, protected_ts_race, protected_ts_sex):
    # realList = {}
    # predList = {}
    realList = []
    predList = []
    for index, row in test_dataset.iterrows():
        idName = row["indexA"]
        datapoint = np.array(row["A"])
        real_score = row["Score"]
        datapoint = np.expand_dims(datapoint, axis=0)
        pred_score = dual_encoder.score(datapoint)
        pred_score = pred_score.numpy()[0][0].item()
        # realList[idName] = real_score
        # predList[idName] = pred_score
        realList.append({"id": idName, "Score": real_score})
        predList.append({"id": idName, "Score": pred_score})
    realList = pd.DataFrame(realList)
    predList = pd.DataFrame(predList)

    m_comp = metrics.Metrics(realList["Score"], predList["Score"])

    [spearmanr, sp_pvalue] = stats.spearmanr(realList["Score"], predList["Score"])
    [pearsonr, p_pvalue] = stats.pearsonr(realList["Score"], predList["Score"])

    # realList.sort_values(by=['Score'], inplace=True)
    # predList.sort_values(by=['Score'], inplace=True)
    # realList = realList.reset_index()
    # predList = predList.reset_index()
    return spearmanr, sp_pvalue, pearsonr, p_pvalue, m_comp.MI_con_info(protected_ts_race), m_comp.MI_con_info(
        protected_ts_sex)


def evaluateLists(realList, predList):
    realList = realList["id"].tolist()
    predList = predList["id"].tolist()
    ln = len(realList)
    diff = 0
    sum_d = 0
    for i in range(ln):
        id = realList[i]
        j = predList.index(id)
        diff += (abs(i - j))
        sum_d += ((i - j) * (i - j))
    spearman_corr = 1 - ((6 * sum_d) / (ln * ((ln * ln) - 1)))
    spearman_corr = round(spearman_corr, 3)
    avg_diff = round(diff / ln, 3)
    return avg_diff, spearman_corr


def comparabilityExperiment(protected_ts_race, protected_ts_sex, protected_ts_AB_race, protected_ts_AB_sex,
                            dataName="FaceImage", train_val=None, test=None, testList=None, dataList=None, height=250,
                            width=250):
    np.random.shuffle(train_val.values)
    train = train_val.head(int((len(train_val) * 0.8)))
    y_true = train["Label"].tolist()
    val = train_val.drop(train.index)
    np.random.shuffle(test.values)

    print("Training...")
    dual_encoder = train_model(train=train, val=val, y_true=y_true, shared=True, height=height, width=width)
    print("Finished training.")
    print("Testing...")
    recall, precision, F1, accuracy, AOD_race = test_model(test, dual_encoder, protected_ts_AB_race)
    recall, precision, F1, accuracy, AOD_sex = test_model(test, dual_encoder, protected_ts_AB_sex)

    print(recall, precision, F1, accuracy)

    spearmanr, sp_pvalue, pearsonr, p_pvalue, MI_encoder_race, MI_encoder_sex = generateLists(testList, dual_encoder,
                                                                                              protected_ts_race,
                                                                                              protected_ts_sex)
    # realList.to_csv("../../Results/Real Order " + dataName + ".csv", index=False)
    # predList.to_csv("../../Results/Prediction Order " + dataName + ".csv", index=False)
    # avg_diff, spearman_corr = evaluateLists(realList, predList)
    # print(avg_diff, spearman_corr)

    # realList, predList = generateLists(dataList, dual_encoder)
    # realList.to_csv("../../Results/Real Order " + dataName + " Full.csv", index=False)
    # predList.to_csv("../../Results/Prediction Order " + dataName + " Full.csv", index=False)
    # avg_diff_full = evaluateLists(realList, predList)
    # print(avg_diff_full)

    return recall, precision, F1, accuracy, AOD_race, AOD_sex, spearmanr, sp_pvalue, pearsonr, p_pvalue, MI_encoder_race, MI_encoder_sex


def regressionExperiment(train_val,
                         test,
                         comp_test, protected_ts_race, protected_ts_sex,
                         height=250,
                         width=250, col="Average"):
    protected_reg = []

    train_val['pixels'] = train_val['Filename'].apply(DataProcessing.retrievePixels)
    test['pixels'] = test['Filename'].apply(DataProcessing.retrievePixels)
    train = train_val.head(int((len(train_val) * 0.8)))
    val = train_val.drop(train.index)

    features_tr = np.array([pixel for pixel in train['pixels']]) / 255.0
    features_val = np.array([pixel for pixel in val['pixels']]) / 255.0
    features_ts = np.array([pixel for pixel in test['pixels']]) / 255.0

    # for file in test['Filename']:
    #     if file[1] == 'M':
    #         protected_reg.append(1)
    #     else:
    #         protected_reg.append(0)

    X_train = features_tr
    X_val = features_val
    X_test = features_ts

    y_train = np.array(train[col])
    y_val = np.array(val[col])

    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # with strategy.scope():
    model = vgg_pre.VGG_Pre()

    model.fit(X_train, train[col], X_val, val[col])

    realList = comp_test["Label"].to_list()
    predList = []

    for index, row in comp_test.iterrows():
        indexA = row["A"]
        indexB = row["B"]
        indexA = np.expand_dims(indexA, axis=0)
        indexB = np.expand_dims(indexB, axis=0)
        pred_score_A = model.decision_function(indexA)
        pred_score_B = model.decision_function(indexB)
        if pred_score_A > pred_score_B:
            predList.append(1)
        else:
            predList.append(-1)

    m_comp = metrics.Metrics(realList, predList)

    preds = model.decision_function(X_test).flatten()
    m = metrics.Metrics(test[col], preds)

    return m.mse(), m.r2(), m.pearsonr_coefficient(), m.pearsonr_value(), m.spearmanr_coefficient(), m.spearmanr_value(), m.MI_con_info(
        protected_ts_race), m.MI_con_info(protected_ts_sex), m.r_sep(protected_ts_race), m.r_sep(
        protected_ts_sex), m_comp.accuracy(), m_comp.f1(), m_comp.precision(), m_comp.recall()
    # result["R2"] = m.r2()
    # result["P Coefficient"] = m.p_coefficient()
    # result["P Value"] = m.p_value()
    #
    # protected = ['Sex']
    #
    # for A in protected:
    #     result["MI_" + A] = "%.2f" % (m.MI(protected_reg))
    #     result["R_sep_" + A] = "%.2f" % (m.r_sep(protected_reg))
