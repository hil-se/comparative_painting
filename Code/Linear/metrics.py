import math

import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.stats import t, norm, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler


class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = np.array(y)
        self.y_pred = np.array(y_pred)

    def mse(self):
        return sklearn.metrics.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    def accuracy(self):
        return sklearn.metrics.accuracy_score(self.y, self.y_pred)

    def f1(self):
        return sklearn.metrics.f1_score(self.y, self.y_pred)

    def precision(self):
        return sklearn.metrics.precision_score(self.y, self.y_pred)

    def recall(self):
        return sklearn.metrics.recall_score(self.y, self.y_pred)

    def r2(self):
        return sklearn.metrics.r2_score(self.y, self.y_pred)

    def pearsonr_coefficient(self):
        return pearsonr(self.y, self.y_pred).statistic

    def pearsonr_value(self):
        return pearsonr(self.y, self.y_pred).pvalue

    def spearmanr_coefficient(self):
        return spearmanr(self.y, self.y_pred).statistic

    def spearmanr_value(self):
        return spearmanr(self.y, self.y_pred).pvalue

    def RBD(self, s):
        # s is an array of numerical values of a sensitive attribute
        if len(np.unique(s)) == 2:
            error = np.array(self.y_pred) - np.array(self.y)
            bias = {}
            bias[1] = error[np.where(np.array(s) == 1)[0]]
            bias[0] = error[np.where(np.array(s) == 0)[0]]
            bias_diff = np.mean(bias[1]) - np.mean(bias[0])
        else:
            bias_diff = 0.0
            n = 0
            for i in range(len(self.y)):
                for j in range(len(self.y)):
                    if np.array(s)[i] - np.array(s)[j] > 0:
                        diff_pred = self.y_pred[i] - self.y_pred[j]
                        diff_true = self.y[i] - self.y[j]
                        n += 1
                        bias_diff += diff_pred - diff_true
            bias_diff = bias_diff / n
        sigma = np.std(self.y_pred - self.y, ddof=1)
        if sigma:
            bias_diff = bias_diff / sigma
        else:
            bias_diff = 0.0
        return bias_diff

    def RBT(self, s):
        # s is an array of numerical values of a sensitive attribute
        if len(np.unique(s)) == 2:
            error = np.array(self.y_pred) - np.array(self.y)
            bias = {}
            bias[1] = error[np.where(np.array(s) == 1)[0]]
            bias[0] = error[np.where(np.array(s) == 0)[0]]
            bias_diff = np.mean(bias[1]) - np.mean(bias[0])
            var1 = np.var(bias[1], ddof=1)
            var0 = np.var(bias[0], ddof=1)
            var = var1 / len(bias[1]) + var0 / len(bias[0])
            if var > 0:
                bias_diff = bias_diff / np.sqrt(var)
                dof = var ** 2 / ((var1 / len(bias[1])) ** 2 / (len(bias[1]) - 1) + (var0 / len(bias[0])) ** 2 / (
                        len(bias[0]) - 1))
            else:
                bias_diff = 0.0
                dof = 1
        else:
            bias_diff = 0.0
            n = 0
            for i in range(len(self.y)):
                for j in range(len(self.y)):
                    if np.array(s)[i] - np.array(s)[j] > 0:
                        diff_pred = self.y_pred[i] - self.y_pred[j]
                        diff_true = self.y[i] - self.y[j]
                        n += 1
                        bias_diff += diff_pred - diff_true
            bias_diff = bias_diff / n
            sigma = np.std(self.y_pred - self.y, ddof=1)
            if sigma:
                bias_diff = bias_diff * np.sqrt(len(s)) / sigma
            else:
                bias_diff = 0.0
            dof = len(s) - 1
        p = t.sf(np.abs(bias_diff), dof)
        return p

    def r_sep(self, s):

        joint = pd.DataFrame({'y': self.y, 'y_pred': self.y_pred}, columns=['y', 'y_pred'])
        margin = self.y.reshape(-1, 1)
        model_joint = LogisticRegression().fit(joint, s)
        model_margin = LogisticRegression().fit(margin, s)

        prob_joint = model_joint.predict_proba(joint)[:, 1]
        prob_margin = model_margin.predict_proba(margin)[:, 1]
        ratio = 0

        for i in range(len(s)):
            t = (prob_joint[i] / (1 - prob_joint[i])) * ((1 - prob_margin[i]) / prob_margin[i])
            ratio = ratio + t
        ratio = ratio / len(s)
        return ratio

    def MI(self, s):

        joint = pd.DataFrame({'y': self.y, 'y_pred': self.y_pred}, columns=['y', 'y_pred'])
        margin = self.y.reshape(-1, 1)
        model_joint = LogisticRegression().fit(joint, s)
        model_margin = LogisticRegression().fit(margin, s)

        prob_joint = model_joint.predict_proba(joint)
        prob_margin = model_margin.predict_proba(margin)
        Info = 0
        Entropy = 0

        for i in range(len(s)):
            Info = Info + math.log(prob_joint[i][s[i]] / prob_margin[i][s[i]])
            Entropy = Entropy + math.log(prob_margin[i][s[i]])

        MI = Info / (-Entropy)
        return MI

    def MI_con(self, s):

        joint = pd.DataFrame({'y': self.y, 'y_pred': self.y_pred}, columns=['y', 'y_pred'])
        margin = self.y.reshape(-1, 1)

        model_joint = LinearRegression().fit(joint, s)
        model_margin = LinearRegression().fit(margin, s)

        pred_joint = model_joint.predict(joint)
        pred_margin = model_margin.predict(margin)

        # plt.scatter(joint['y_pred'], s)
        # plt.xlabel('y_pred')
        # plt.ylabel('s')
        # plt.show()

        # plt.scatter(joint['y'], s)
        # plt.xlabel('y')
        # plt.ylabel('s')
        # plt.show()

        # plt.scatter(margin, s)
        # plt.plot(margin, pred_margin, color='red')
        # plt.xlabel('y')
        # plt.ylabel('s')
        # plt.show()

        # score = model_joint.score(joint, s)

        # resid = pred_joint - s
        # mu, std = norm.fit(resid)
        # plt.scatter(pred_joint, resid)
        # plt.xlabel('pred_joint')
        # plt.ylabel('Residual Error of Regression')
        # plt.show()

        # name = ['Jarque-Bera test', 'Chi-squared(2) p-value', 'Skewness', 'Kurtosis']
        # test = sms.jarque_bera(resid)
        # print(lzip(name, test))

        # plt.hist(resid, bins=50)
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter3D(joint['y'], joint['y_pred'], s, color="green")
        # ax.scatter3D(joint['y'], joint['y_pred'], pred_joint, color="red")
        # ax.set_xlabel('y')
        # ax.set_ylabel('y_pred')
        # ax.set_zlabel('s')
        # plt.show()

        # N = len(joint)
        # p = len(joint.columns) + 1  # plus one because LinearRegression adds an intercept term
        #
        # X_with_intercept = np.empty(shape=(N, p), dtype=float)
        # X_with_intercept[:, 0] = 1
        # X_with_intercept[:, 1:p] = joint.values
        #
        # keys = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
        # results = het_white(resid, X_with_intercept)
        # print(lzip(keys, results))
        rse_joint = np.std(pred_joint - s)
        rse_margin = np.std(pred_margin - s)

        pdf_joint = norm.pdf(s, pred_joint, rse_joint)
        pdf_margin = norm.pdf(s, pred_margin, rse_margin)

        Info = 0
        Entropy = 0

        for i in range(len(s)):
            Info = Info + math.log(
                pdf_joint[i] / pdf_margin[i])
            Entropy = Entropy + math.log(pdf_margin[i])

        MI = Info / (-Entropy)
        return MI

    def MI_con_scaled(self, s):

        joint = pd.DataFrame({'y': self.y, 'y_pred': self.y_pred}, columns=['y', 'y_pred'])
        margin = self.y.reshape(-1, 1)

        model_joint = LinearRegression().fit(joint, s)
        model_margin = LinearRegression().fit(margin, s)

        pred_joint = model_joint.predict(joint)
        pred_margin = model_margin.predict(margin)

        rse_joint = np.std(pred_joint - s)
        rse_margin = np.std(pred_margin - s)

        pdf_joint = norm.pdf(s, pred_joint, rse_joint)
        pdf_margin = norm.pdf(s, pred_margin, rse_margin)

        concat_pdf = np.concatenate((pdf_joint, pdf_margin))

        scaler = MinMaxScaler(feature_range=(0.01, 0.99))

        scaled_concat_pdf = scaler.fit_transform(concat_pdf.reshape(-1, 1))

        scaled_joint_pdf, scaled_pdf_margin = np.array_split(scaled_concat_pdf, 2)

        Info = 0
        Entropy = 0

        for i in range(len(s)):
            Info = Info + math.log(
                scaled_joint_pdf[i] / scaled_pdf_margin[i])
            Entropy = Entropy + math.log(scaled_pdf_margin[i])

        MI = Info / (-Entropy)
        return MI

    def MI_con_info(self, s):

        joint = pd.DataFrame({'y': self.y, 'y_pred': self.y_pred}, columns=['y', 'y_pred'])
        margin = self.y.reshape(-1, 1)

        model_joint = LinearRegression().fit(joint, s)
        model_margin = LinearRegression().fit(margin, s)

        pred_joint = model_joint.predict(joint)
        pred_margin = model_margin.predict(margin)

        rse_joint = np.std(pred_joint - s)
        rse_margin = np.std(pred_margin - s)

        pdf_joint = norm.pdf(s, pred_joint, rse_joint)
        pdf_margin = norm.pdf(s, pred_margin, rse_margin)

        Info = 0

        for i in range(len(s)):
            Info = Info + math.log(
                pdf_joint[i] / pdf_margin[i])

        MI = Info / len(s)
        return MI
