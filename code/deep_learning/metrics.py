"""Evaluation metrics for art evaluation prediction models.

This module provides a Metrics class that computes standard regression and
correlation metrics used to evaluate model performance in the IEEE Access
paper "Modeling Art Evaluations from Comparative Judgments." The metrics
include MAE, MSE, R-squared, Pearson correlation, and Spearman rank
correlation.

These metrics are used by both the direct regression (regression.py) and
comparative pairwise (comparative.py) models to assess how well predicted
scores match human rater judgments.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
import sklearn.metrics as m


class Metrics:
    """Compute evaluation metrics between true and predicted scores.

    Stores ground-truth and predicted values and provides methods to
    compute various error and correlation metrics. Used after model
    inference to evaluate prediction quality on the test set.

    Attributes:
        y (array-like): Ground-truth values (human rater scores).
        y_pred (array-like): Model-predicted values.
    """

    def __init__(self, y, y_pred):
        """Initialize Metrics with ground-truth and predicted values.

        Args:
            y (array-like): Ground-truth values (e.g., human beauty or
                liking ratings).
            y_pred (array-like): Predicted values from the model.
        """
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = y
        self.y_pred = y_pred

    def mse(self):
        """Compute Mean Squared Error between true and predicted values.

        Returns:
            float: Mean squared error.
        """
        return m.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        """Compute Mean Absolute Error between true and predicted values.

        Manually computed as sum(|y - y_pred|) / n rather than using
        sklearn, to maintain consistency with the paper's reporting.

        Returns:
            float: Mean absolute error.
        """
        return np.sum(np.abs(np.array(self.y) - np.array(self.y_pred)))/len(self.y)
        # return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    def r2(self):
        """Compute R-squared (coefficient of determination).

        Measures the proportion of variance in the true values that is
        explained by the predicted values. A value of 1.0 indicates
        perfect prediction; negative values indicate worse than the mean.

        Returns:
            float: R-squared score.
        """
        return m.r2_score(self.y, self.y_pred)

    def pearsonr(self):
        """Compute Pearson correlation coefficient between predictions and truth.

        Measures linear correlation between predicted and true values.
        Reported as 'rho' in the paper's results tables.

        Returns:
            scipy.stats.PearsonRResult: Result object with .statistic
                (correlation coefficient) and .pvalue attributes.
        """
        return pearsonr(self.y_pred, self.y)

    def spearmanr(self):
        """Compute Spearman rank correlation between predictions and truth.

        Measures monotonic relationship between predicted and true values,
        based on ranks rather than raw values. Reported as 'rs' in the
        paper's results tables. Particularly relevant for the comparative
        model since it learns rankings rather than absolute scores.

        Returns:
            scipy.stats.SignificanceResult: Result object with .statistic
                (correlation coefficient) and .pvalue attributes.
        """
        return spearmanr(self.y_pred, self.y)
