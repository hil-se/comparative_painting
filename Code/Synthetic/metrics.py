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
        return pearsonr(self.y, self.y_pred)[0]

    def pearsonr_value(self):
        return pearsonr(self.y, self.y_pred)[1]

    def spearmanr_coefficient(self):
        return spearmanr(self.y, self.y_pred)[0]

    def spearmanr_value(self):
        return spearmanr(self.y, self.y_pred)[1]

    def confusion(self, y, y_pred):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(y)):
            if y[i] > 0:
                if y_pred[i] > 0:
                    tp += 1
                else:
                    fn += 1
            else:
                if y_pred[i] > 0:
                    fp += 1
                else:
                    tn += 1
        return tp, fp, tn, fn

    def EOD(self, s):
        # True positive rate (TPR)
        y0 = self.y[s == 0]
        y0_pred = self.y_pred[s == 0]
        y1 = self.y[s == 1]
        y1_pred = self.y_pred[s == 1]

        tp, fp, tn, fn = self.confusion(y0, y0_pred)
        op0 = float(tp) / (tp + fn)
        tp, fp, tn, fn = self.confusion(y1, y1_pred)
        op1 = float(tp) / (tp + fn)
        return op1 - op0

    def AOD(self, s):
        # equal TPR and equal FPR
        y0 = self.y[s == 0]
        y0_pred = self.y_pred[s == 0]
        y1 = self.y[s == 1]
        y1_pred = self.y_pred[s == 1]

        tp, fp, tn, fn = self.confusion(y0, y0_pred)
        od0 = float(tp) / (tp + fn) + float(fp) / (fp + tn)
        tp, fp, tn, fn = self.confusion(y1, y1_pred)
        od1 = float(tp) / (tp + fn) + float(fp) / (fp + tn)
        return (od1 - od0) / 2

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

    def MI_b(self, s):

        y0 = self.y[s == 0]
        y0_pred = self.y_pred[s == 0]
        y1 = self.y[s == 1]
        y1_pred = self.y_pred[s == 1]

        tp0, fp0, tn0, fn0 = self.confusion(y0, y0_pred)
        tp1, fp1, tn1, fn1 = self.confusion(y1, y1_pred)
        def ediff(n1, d1, n0, d0):
            return np.log(n0/(n0+d0))*n0 + np.log(n1/(n1+d1))*n1 - (n0+n1)*np.log((n0+n1)/(n0+n1+d0+d1))

        MI = ediff(tp1, fn1, tp0, fn0) + ediff(fp1, tn1, fp0, tn0) + ediff(tn1, fp1, tn0, fp0) + ediff(fn1, tp1, fn0, tp0)
        return MI / len(s)

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

    def AOD_comp(self, s):

        t = n = tp = fp = tn = fn = 0

        for i, row in s.iterrows():
            if row['A'] > row['B']:
                if self.y[i] == 1:
                    t += 1
                    if self.y_pred[i] == 1:
                        tp += 1
                    if self.y_pred[i] == -1:
                        fn += 1
                elif self.y[i] == -1:
                    n += 1
                    if self.y_pred[i] == 1:
                        fp += 1
                    if self.y_pred[i] == -1:
                        tn += 1

            elif row['A'] < row['B']:
                if self.y[i] == -1:
                    t += 1
                    if self.y_pred[i] == -1:
                        tp += 1
                    if self.y_pred[i] == 1:
                        fn += 1
                elif self.y[i] == 1:
                    n += 1
                    if self.y_pred[i] == -1:
                        fp += 1
                    if self.y_pred[i] == 1:
                        tn += 1

        tpr = tp / t
        tnr = tn / n
        fpr = fp / n
        fnr = fn / t
        aod = (tpr + fpr - tnr - fnr) / 2

        return aod

    def Within_comp(self, s):

        t1 = n1 = tp1 = fp1 = tn1 = fn1 = 0
        t0 = n0 = tp0 = fp0 = tn0 = fn0 = 0

        for i, row in s.iterrows():
            if row['A'] == row['B'] == 1:
                if self.y[i] == 1:
                    t1 += 1
                    if self.y_pred[i] == 1:
                        tp1 += 1
                    if self.y_pred[i] == -1:
                        fn1 += 1
                elif self.y[i] == -1:
                    n1 += 1
                    if self.y_pred[i] == 1:
                        fp1 += 1
                    if self.y_pred[i] == -1:
                        tn1 += 1

            elif row['A'] == row['B'] == 0:
                if self.y[i] == 1:
                    t0 += 1
                    if self.y_pred[i] == 1:
                        tp0 += 1
                    if self.y_pred[i] == -1:
                        fn0 += 1
                elif self.y[i] == -1:
                    n0 += 1
                    if self.y_pred[i] == 1:
                        fp0 += 1
                    if self.y_pred[i] == -1:
                        tn0 += 1

        tpr1 = (tp1+tn1) / (t1+n1)
        fpr1 = (fp1+fn1) / (t1+n1)
        tpr0 = (tp0+tn0) / (t0+n0)
        fpr0 = (fp0+fn0) / (t0+n0)
        within = (tpr1 - fpr1 - tpr0 + fpr0) / 2

        return within

    def Sep_comp(self, s):
        return np.sqrt(self.Within_comp(s)**2+self.AOD_comp(s)**2)

    def gAOD(self, s):
        # s is an array of numerical values of a sensitive attribute
        t = n = tp = fp = tn = fn = 0
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if s[i] - s[j] > 0:
                    if self.y[i] - self.y[j] > 0:
                        t += 1
                        if self.y_pred[i] > self.y_pred[j]:
                            tp += 1
                        if self.y_pred[i] < self.y_pred[j]:
                            fn += 1
                    elif self.y[j] - self.y[i] > 0:
                        n += 1
                        if self.y_pred[i] > self.y_pred[j]:
                            fp += 1
                        elif self.y_pred[i] < self.y_pred[j]:
                            tn += 1

        tpr = tp / t
        tnr = tn / n
        fpr = fp / n
        fnr = fn / t
        aod = (tpr + fpr - tnr - fnr) / 2
        return aod

    def gWithin(self, s):
        # s is an array of numerical values of a sensitive attribute
        t1 = n1 = tp1 = fp1 = tn1 = fn1 = 0
        t0 = n0 = tp0 = fp0 = tn0 = fn0 = 0
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if s[i] == s[j] == 1:
                    if self.y[i] - self.y[j] > 0:
                        t1 += 1
                        if self.y_pred[i] > self.y_pred[j]:
                            tp1 += 1
                        if self.y_pred[i] < self.y_pred[j]:
                            fn1 += 1
                    elif self.y[j] - self.y[i] > 0:
                        n1 += 1
                        if self.y_pred[i] > self.y_pred[j]:
                            fp1 += 1
                        elif self.y_pred[i] < self.y_pred[j]:
                            tn1 += 1
                elif s[i] == s[j] == 0:
                    if self.y[i] - self.y[j] > 0:
                        t0 += 1
                        if self.y_pred[i] > self.y_pred[j]:
                            tp0 += 1
                        if self.y_pred[i] < self.y_pred[j]:
                            fn0 += 1
                    elif self.y[j] - self.y[i] > 0:
                        n0 += 1
                        if self.y_pred[i] > self.y_pred[j]:
                            fp0 += 1
                        elif self.y_pred[i] < self.y_pred[j]:
                            tn0 += 1

        tpr1 = (tp1 + tn1) / (t1 + n1)
        fpr1 = (fp1 + fn1) / (t1 + n1)
        tpr0 = (tp0 + tn0) / (t0 + n0)
        fpr0 = (fp0 + fn0) / (t0 + n0)
        within = (tpr1 - fpr1 - tpr0 + fpr0) / 2

        return within

    def gSep(self, s):
        return np.sqrt(self.gWithin(s) ** 2 + self.gAOD(s) ** 2)

    def MI_comp(self, s):
        t = n = tp = fp = tn = fn = 0
        t11 = n11 = tp11 = fp11 = tn11 = fn11 = 0
        t10 = n10 = tp10 = fp10 = tn10 = fn10 = 0
        t01 = n01 = tp01 = fp01 = tn01 = fn01 = 0
        t00 = n00 =tp00 = fp00 = tn00 = fn00 = 0

        for i, row in s.iterrows():
            if self.y[i] == 1:
                t += 1
                if row['A'] == row['B'] == 1:
                    t11 += 1
                if row['A'] == row['B'] == 0:
                    t00 += 1
                if row['A'] == 1 and row['B'] == 0:
                    t10 += 1
                if row['A'] == 0 and row['B'] == 1:
                    t01 += 1
                if self.y_pred[i] == 1:
                    tp += 1
                    if row['A'] == row['B'] == 1:
                        tp11 += 1
                    if row['A'] == row['B'] == 0:
                        tp00 += 1
                    if row['A'] == 1 and row['B'] == 0:
                        tp10 += 1
                    if row['A'] == 0 and row['B'] == 1:
                        tp01 += 1
                if self.y_pred[i] == -1:
                    fn += 1
                    if row['A'] == row['B'] == 1:
                        fn11 += 1
                    if row['A'] == row['B'] == 0:
                        fn00 += 1
                    if row['A'] == 1 and row['B'] == 0:
                        fn10 += 1
                    if row['A'] == 0 and row['B'] == 1:
                        fn01 += 1
            elif self.y[i] == -1:
                n += 1
                if row['A'] == row['B'] == 1:
                    n11 += 1
                if row['A'] == row['B'] == 0:
                    n00 += 1
                if row['A'] == 1 and row['B'] == 0:
                    n10 += 1
                if row['A'] == 0 and row['B'] == 1:
                    n01 += 1
                if self.y_pred[i] == 1:
                    fp += 1
                    if row['A'] == row['B'] == 1:
                        fp11 += 1
                    if row['A'] == row['B'] == 0:
                        fp00 += 1
                    if row['A'] == 1 and row['B'] == 0:
                        fp10 += 1
                    if row['A'] == 0 and row['B'] == 1:
                        fp01 += 1
                if self.y_pred[i] == -1:
                    tn += 1
                    if row['A'] == row['B'] == 1:
                        tn11 += 1
                    if row['A'] == row['B'] == 0:
                        tn00 += 1
                    if row['A'] == 1 and row['B'] == 0:
                        tn10 += 1
                    if row['A'] == 0 and row['B'] == 1:
                        tn01 += 1
        tp = tp00 + tp01 + tp10 + tp11
        fn = fn00 + fn01 + fn10 + fn11
        fp = fp00 + fp01 + fp10 + fp11
        tn = tn00 + tn01 + tn10 + tn11

        mitp00 = np.log(tp00 / tp / (t00 / t)) * tp00 / len(s)
        mifn00 = np.log(fn00 / fn / (t00 / t)) * fn00 / len(s)
        mifp00 = np.log(fp00 / fp / (n00 / n)) * fp00 / len(s)
        mitn00 = np.log(tn00 / tn / (n00 / n)) * tn00 / len(s)
        mi00 = mitp00 + mifn00 + mifp00 + mitn00

        mitp01 = np.log(tp01 / tp / (t01 / t)) * tp01 / len(s)
        mifn01 = np.log(fn01 / fn / (t01 / t)) * fn01 / len(s)
        mifp01 = np.log(fp01 / fp / (n01 / n)) * fp01 / len(s)
        mitn01 = np.log(tn01 / tn / (n01 / n)) * tn01 / len(s)
        mi01 = mitp01 + mifn01 + mifp01 + mitn01

        mitp10 = np.log(tp10 / tp / (t10 / t)) * tp10 / len(s)
        mifn10 = np.log(fn10 / fn / (t10 / t)) * fn10 / len(s)
        mifp10 = np.log(fp10 / fp / (n10 / n)) * fp10 / len(s)
        mitn10 = np.log(tn10 / tn / (n10 / n)) * tn10 / len(s)
        mi10 = mitp10 + mifn10 + mifp10 + mitn10

        mitp11 = np.log(tp11 / tp / (t11 / t)) * tp11 / len(s)
        mifn11 = np.log(fn11 / fn / (t11 / t)) * fn11 / len(s)
        mifp11 = np.log(fp11 / fp / (n11 / n)) * fp11 / len(s)
        mitn11 = np.log(tn11 / tn / (n11 / n)) * tn11 / len(s)
        mi11 = mitp11 + mifn11 + mifp11 + mitn11
        mi = mi00 + mi01 + mi10 + mi11
        return mi

    def MI_comp2(self, s):
        t = n = tp = fp = tn = fn = 0
        t11 = n11 = tp11 = fp11 = tn11 = fn11 = 0
        t10 = n10 = tp10 = fp10 = tn10 = fn10 = 0
        t01 = n01 = tp01 = fp01 = tn01 = fn01 = 0
        t00 = n00 =tp00 = fp00 = tn00 = fn00 = 0

        for i, row in s.iterrows():
            if self.y[i] == 1:
                t += 1
                if row['A'] == row['B'] == 1:
                    t11 += 1
                if row['A'] == row['B'] == 0:
                    t00 += 1
                if row['A'] == 1 and row['B'] == 0:
                    t10 += 1
                if row['A'] == 0 and row['B'] == 1:
                    t01 += 1
                if self.y_pred[i] == 1:
                    tp += 1
                    if row['A'] == row['B'] == 1:
                        tp11 += 1
                    if row['A'] == row['B'] == 0:
                        tp00 += 1
                    if row['A'] == 1 and row['B'] == 0:
                        tp10 += 1
                    if row['A'] == 0 and row['B'] == 1:
                        tp01 += 1
                if self.y_pred[i] == -1:
                    fn += 1
                    if row['A'] == row['B'] == 1:
                        fn11 += 1
                    if row['A'] == row['B'] == 0:
                        fn00 += 1
                    if row['A'] == 1 and row['B'] == 0:
                        fn10 += 1
                    if row['A'] == 0 and row['B'] == 1:
                        fn01 += 1
            elif self.y[i] == -1:
                n += 1
                if row['A'] == row['B'] == 1:
                    n11 += 1
                if row['A'] == row['B'] == 0:
                    n00 += 1
                if row['A'] == 1 and row['B'] == 0:
                    n10 += 1
                if row['A'] == 0 and row['B'] == 1:
                    n01 += 1
                if self.y_pred[i] == 1:
                    fp += 1
                    if row['A'] == row['B'] == 1:
                        fp11 += 1
                    if row['A'] == row['B'] == 0:
                        fp00 += 1
                    if row['A'] == 1 and row['B'] == 0:
                        fp10 += 1
                    if row['A'] == 0 and row['B'] == 1:
                        fp01 += 1
                if self.y_pred[i] == -1:
                    tn += 1
                    if row['A'] == row['B'] == 1:
                        tn11 += 1
                    if row['A'] == row['B'] == 0:
                        tn00 += 1
                    if row['A'] == 1 and row['B'] == 0:
                        tn10 += 1
                    if row['A'] == 0 and row['B'] == 1:
                        tn01 += 1

        mitp00 = np.log(tp00 / t00 / (tp / t)) * tp00 / len(s)
        mifn00 = np.log(fn00 / t00 / (fn / t)) * fn00 / len(s)
        mifp00 = np.log(fp00 / n00 / (fp / n)) * fp00 / len(s)
        mitn00 = np.log(tn00 / n00 / (tn / n)) * tn00 / len(s)
        mi00 = mitp00 + mifn00 + mifp00 + mitn00

        mitp01 = np.log(tp01 / t01 / (tp / t)) * tp01 / len(s)
        mifn01 = np.log(fn01 / t01 / (fn / t)) * fn01 / len(s)
        mifp01 = np.log(fp01 / n01 / (fp / n)) * fp01 / len(s)
        mitn01 = np.log(tn01 / n01 / (tn / n)) * tn01 / len(s)
        mi01 = mitp01 + mifn01 + mifp01 + mitn01

        mitp10 = np.log(tp10 / t10 / (tp / t)) * tp10 / len(s)
        mifn10 = np.log(fn10 / t10 / (fn / t)) * fn10 / len(s)
        mifp10 = np.log(fp10 / n10 / (fp / n)) * fp10 / len(s)
        mitn10 = np.log(tn10 / n10 / (tn / n)) * tn10 / len(s)
        mi10 = mitp10 + mifn10 + mifp10 + mitn10

        mitp11 = np.log(tp11 / t11 / (tp / t)) * tp11 / len(s)
        mifn11 = np.log(fn11 / t11 / (fn / t)) * fn11 / len(s)
        mifp11 = np.log(fp11 / n11 / (fp / n)) * fp11 / len(s)
        mitn11 = np.log(tn11 / n11 / (tn / n)) * tn11 / len(s)
        mi11 = mitp11 + mifn11 + mifp11 + mitn11
        mi = mi00 + mi01 + mi10 + mi11
        return mi
