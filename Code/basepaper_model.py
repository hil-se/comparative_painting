import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr, spearmanr

# Load dataset (assuming CSV format with relevant columns)
data = pd.read_csv("Data/PaintingDataMeans1.csv")
data.replace(["#NULL!", np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Filter only representational paintings
data = data[data["Representational"] == 1].copy()

# Define predictors and target variables
objective_predictors = ["HueSD", "SaturationSD", "Brightness", "BrightnessSD", 
                        "ColourComponent", "Entropy", "StraightEdgeDensity", "NonStraightEdgeDensity",
                        "Horizontal_Symmetry", "Vertical_Symmetry"]
targets = ["Liking_M"]

# Standardize objective predictors for consistency
scaler = StandardScaler()
data[objective_predictors] = scaler.fit_transform(data[objective_predictors])

# Define stepwise regression function
def stepwise_selection(X, y, threshold_in=0.05, threshold_out=0.1):
    included = []
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        if excluded:
            new_pval = pd.Series(index=excluded, dtype=float)
            for col in excluded:
                model = sm.OLS(y, sm.add_constant(X[included + [col]])).fit()
                new_pval[col] = model.pvalues[col]

            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True

        # Backward step
        if included:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True

        if not changed:
            break
    return included

# Run stepwise regression for beauty ratings
X = data[objective_predictors]
y_beauty = data["Liking_M"]
selected_features_beauty = stepwise_selection(X, y_beauty)

# Check if any features were selected
if not selected_features_beauty:
    print("No features selected by stepwise regression.")
else:
    # Fit final regression model
    final_model_beauty = sm.OLS(y_beauty, sm.add_constant(X[selected_features_beauty])).fit()
    print(final_model_beauty.summary())

    # Calculate Variance Inflation Factor (VIF) for multicollinearity check
    X_filtered = X[selected_features_beauty]
    if X_filtered.shape[1] > 1:  # VIF is meaningful only for multiple predictors
        vif_data = pd.DataFrame()
        vif_data["Feature"] = selected_features_beauty
        vif_data["VIF"] = [variance_inflation_factor(X_filtered.values, i) for i in range(X_filtered.shape[1])]
        print(vif_data)
    else:
        print("Skipping VIF calculation due to insufficient features.")

    # Print model performance metrics
    adjusted_r2 = final_model_beauty.rsquared_adj
    bic = final_model_beauty.bic
    print(f"Adjusted R²: {adjusted_r2:.4f}")
    print(f"Bayesian Information Criterion (BIC): {bic:.4f}")

    # Compute Pearson and Spearman correlations
    pearson_corr, pearson_pval = pearsonr(y_beauty, final_model_beauty.fittedvalues)
    spearman_corr, spearman_pval = spearmanr(y_beauty, final_model_beauty.fittedvalues)
    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_pval:.4f})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_pval:.4f})")

    # LASSO Regression for Beauty Rating
    #lasso = LassoCV(cv=15).fit(X_filtered, y_beauty)
    #lasso_coefficients = dict(zip(selected_features_beauty, lasso.coef_))
    #print(f"LASSO Selected Coefficients: {lasso_coefficients}")

    # Deep Learning Model for Beauty Prediction
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_beauty, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_filtered.shape[1],)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)

    # Evaluate model performance
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test).flatten()
    deep_pearson_corr, deep_pearson_pval = pearsonr(y_test, predictions)
    deep_spearman_corr, deep_spearman_pval = spearmanr(y_test, predictions)
    deep_adjusted_r2 = 1 - (1 - np.corrcoef(y_test, predictions)[0, 1] ** 2) * (len(y_test) - 1) / (len(y_test) - X_filtered.shape[1] - 1)

    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Deep Learning Adjusted R²: {deep_adjusted_r2:.4f}")
    print(f"Deep Learning Pearson correlation: {deep_pearson_corr:.4f} (p-value: {deep_pearson_pval:.4f})")
    print(f"Deep Learning Spearman correlation: {deep_spearman_corr:.4f} (p-value: {deep_spearman_pval:.4f})")
