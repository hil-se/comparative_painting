import numpy as np
from scipy.stats import pearsonr, spearmanr
import sklearn.metrics as m

class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = y
        self.y_pred = y_pred

    def mse(self):
        return m.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        return np.sum(np.abs(np.array(self.y) - np.array(self.y_pred)))/len(self.y)
        # return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    def r2(self):
        return m.r2_score(self.y, self.y_pred)

    def pearsonr(self):
        return pearsonr(self.y_pred, self.y)

    def spearmanr(self):
        return spearmanr(self.y_pred, self.y)

