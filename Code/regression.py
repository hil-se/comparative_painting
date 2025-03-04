import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Data/Abstract_Data.csv")

# Filter only representational paintings
data = data[data["Representational"] == 0].copy()

# Define predictors and target variable
objective_predictors = ["HueSD", "SaturationSD", "Brightness", "BrightnessSD", 
                        "ColourComponent", "Entropy", "StraightEdgeDensity", "NonStraightEdgeDensity",
                        "Horizontal_Symmetry", "Vertical_Symmetry"]

data.replace("#NULL!", np.nan, inplace=True)
data.dropna(inplace=True)

target = "Liking_M"

# Standardize features
scaler = StandardScaler()
X = data[objective_predictors].values
y = data[target].values.astype(float)
X = scaler.fit_transform(X)

# Data Augmentation
noise_factor = 0.05
X_augmented = np.concatenate([X, X + noise_factor * np.random.randn(*X.shape)])
y_augmented = np.concatenate([y, y + noise_factor * np.std(y) * np.random.randn(*y.shape)])

# Save augmented dataset to CSV
augmented_data = pd.DataFrame(X_augmented, columns=objective_predictors)
augmented_data[target] = y_augmented
augmented_data.to_csv("Data/Augmented_Abstract_Data.csv", index=False)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

def build_encoder(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    return Model(inputs, x)

encoder = build_encoder(X_train.shape[1])

input_data = Input(shape=(X_train.shape[1],))
encoded_data = encoder(input_data)
output = Dense(1, activation='linear')(encoded_data)

regression_model = Model(inputs=input_data, outputs=output)
regression_model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.98, nesterov=True), loss='mse', metrics=['mae'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True, verbose=1)

history = regression_model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test), verbose=1)

y_pred = regression_model.predict(X_test).flatten()

r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - ((1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1))
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
pearson_corr, _ = pearsonr(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)

results = pd.DataFrame({
    "Metric": ["Adjusted R2 Score", "Mean Absolute Error", "Mean Squared Error", "Pearson Correlation", "Spearman Correlation"],
    "Value": [adj_r2, mae, mse, pearson_corr, spearman_corr]
})

results.to_csv("Results/Abstract/regression_results/avg_ratings.csv", index=False)

print("Model evaluation results saved to csv")

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.grid()
plt.show()
