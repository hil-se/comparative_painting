import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

import DualEncoder
import SharedDualEncoder
import metrics


def learn(train_data,
          epochs=100,
          validation_data=None,
          y_true=[],
          patience=10,
          batch_size=256,
          shared=False):
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
    if shared == True:
        encoder = SharedDualEncoder.create_encoder(input_size=train_dataset.element_spec['A'].shape[1])
        dual_encoder = SharedDualEncoder.DualEncoderAll(encoder, y_true=np.array(y_true))
    else:
        encoder_A = DualEncoder.create_encoder(input_size=train_dataset.element_spec['A'].shape[1])
        encoder_B = DualEncoder.create_encoder(input_size=val_dataset.element_spec['A'].shape[1])
        dual_encoder = DualEncoder.DualEncoderAll(encoder_A, encoder_B, y_true=np.array(y_true))
    dual_encoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3))
    # dual_encoder.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.001))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    dual_encoder.fit(
        x=train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[early_stopping],
        verbose=1
    )
    return dual_encoder


def train_model(train, val, y_true, epochs=100, shared=False):
    np.random.shuffle(train.values)
    np.random.shuffle(val.values)
    dual_encoder = learn(train, epochs=epochs, validation_data=val, y_true=y_true, shared=shared)
    return dual_encoder


def test_model(test, dual_encoder):
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

        prediction = prediction.numpy()[0][0].item()

        # Labels: -1, 0, 1
        if prediction < -0.33:
            prediction = -1
        elif prediction > 0.33:
            prediction = 1
        else:
            prediction = 0

        # Labels: -1, 1
        # if prediction<0:
        #     prediction=-1
        # else:
        #     prediction=1

        predictions.append(prediction)
    return evaluate(labels, predictions)
    # return evaluate_accuracy(labels, predictions)


def evaluate_accuracy(y_true, y_pred):
    matches = 0
    ln = len(y_true)
    for i in range(ln):
        if y_pred[i] == y_true[i]:
            matches += 1
    accuracy = matches / ln
    return accuracy


def evaluate(y_true, y_pred):
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
    return recall, precision, F1, accuracy


# def generateLists(test_dataset, dual_encoder):
#     # realList = {}
#     # predList = {}
#     realList = []
#     predList = []
#     for index, row in test_dataset.iterrows():
#         idName = row["indexA"]
#         datapoint = np.array(row["A"])
#         real_score = row["Score"]
#         datapoint = np.expand_dims(datapoint, axis=0)
#         pred_score = dual_encoder.score(datapoint)
#         pred_score = pred_score.numpy()[0][0].item()
#         # realList[idName] = real_score
#         # predList[idName] = pred_score
#         realList.append({"id": idName, "Score": real_score})
#         predList.append({"id": idName, "Score": pred_score})
#     realList = pd.DataFrame(realList)
#     predList = pd.DataFrame(predList)
#     realList.sort_values(by=['Score'], inplace=True)
#     predList.sort_values(by=['Score'], inplace=True)
#     realList = realList.reset_index()
#     predList = predList.reset_index()
#     return realList, predList

def generateLists(test_dataset, dual_encoder):
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
    return spearmanr, sp_pvalue, pearsonr, p_pvalue


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


def explainability(test_dataset, feat_list, dual_encoder):
    res = []
    print("\n\n")
    for index, row in test_dataset.iterrows():
        idName = row["indexA"]
        datapoint = np.array(row["A"])
        real_score = row["Score"]
        datapoint = np.expand_dims(datapoint, axis=0)
        grad = dual_encoder.output_grad(tf.Variable(datapoint))
        pred_score = dual_encoder.score(datapoint)
        pred_score = pred_score.numpy()[0][0].item()

        t_real = {"id": idName}
        t_real["Type"] = "Real (Original data)"
        for i in range(len(feat_list)):
            t_real[feat_list[i]] = (row["A"])[i]
        t_real["Score"] = real_score
        res.append(t_real)

        t_weights = {"id": idName}
        t_weights["Type"] = "Weighted features"
        weighted_feats = (np.multiply(np.array(row["A"]), grad)).tolist()
        for i in range(len(feat_list)):
            t_weights[feat_list[i]] = weighted_feats[i]
        t_weights["Score"] = pred_score
        res.append(t_weights)
        res.append({})
    res = pd.DataFrame(res)
    return res

# def comparabilityExperiment(shared=False, dataName="Boston", testList=None, dataList=None, feat_list=None, epochs=100):
#     r = Reader()
#     r.load(dataName+"_dual_train")
#     train_val = pd.concat([r.A_series, r.B_series, r.labels], axis=1)
#     np.random.shuffle(train_val.values)
#     train = train_val.head(int((len(train_val.index) * 0.7)))
#     y_true = train["Label"].tolist()
#     val = train_val.drop(train.index)
#
#     r = Reader()
#     r.load(dataName+"_dual_test")
#     test = pd.concat([r.A_series, r.B_series, r.labels], axis=1)
#     np.random.shuffle(test.values)
#
#     print("Training...")
#     dual_encoder = train_model(train=train, val=val, y_true=y_true, shared=shared, epochs=epochs)
#     print("Finished training.")
#     print("Testing...")
#
#     # recall, precision, F1, accuracy = test_model(test, dual_encoder)
#     # print(recall, precision, F1, accuracy)
#
#     accuracy = test_model(test, dual_encoder)
#     print(accuracy)
#
#     realList, predList = generateLists(testList, dual_encoder)
#     realList.to_csv("../../Results/Real Order "+dataName+".csv", index=False)
#     predList.to_csv("../../Results/Prediction Order " + dataName + ".csv", index=False)
#     avg_diff, spearman_corr = evaluateLists(realList, predList)
#     print(avg_diff, spearman_corr)
#
#     realList, predList = generateLists(dataList, dual_encoder)
#     realList.to_csv("../../Results/Real Order " + dataName + " Full.csv", index=False)
#     predList.to_csv("../../Results/Prediction Order " + dataName + " Full.csv", index=False)
#     avg_diff_full, spearman_corr_full = evaluateLists(realList, predList)
#     print(avg_diff_full, spearman_corr_full)
#
#     # expln_df = explainability(test_dataset=testList, feat_list=feat_list, dual_encoder=dual_encoder)
#     # expln_df.to_csv("../../Results/Explanations "+dataName+".csv", index=False)
#
#     # return recall, precision, F1, accuracy, avg_diff, avg_diff_full, spearman_corr, spearman_corr_full
#     return accuracy, avg_diff, avg_diff_full, spearman_corr, spearman_corr_full
