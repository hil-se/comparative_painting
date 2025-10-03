import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import scipy.stats as stats
import matplotlib.pyplot as plt

from data_preprocessor_resnet50 import (load_and_preprocess_data, CONFIG, 
                                        build_mlp_encoder, FEATURE_PREFIX)

def build_regression_model_from_encoder(encoder):
    output = Dense(1, activation='linear', name='reg_output')(encoder.output)
    return Model(inputs=encoder.input, outputs=output)

def _train_and_evaluate_regression_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, 
                                         input_dim, condition_name, output_dir):
    encoder = build_mlp_encoder(input_dim)
    model = build_regression_model_from_encoder(encoder)
    
    optimizer = SGD(learning_rate=0.005, momentum=0.9)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=0)
    
    history = model.fit(X_train_scaled, y_train, epochs=300, batch_size=CONFIG['batch_size'],
                        validation_data=(X_val_scaled, y_val), verbose=0, callbacks=[early_stopping])
              
    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss Curve for: {condition_name}')
        plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.legend(); plt.grid(True)
        plot_filename = os.path.join(output_dir, f"loss_plot_{condition_name}.png")
        plt.savefig(plot_filename); plt.close()

    y_pred = model.predict(X_test_scaled).flatten()
    K.clear_session(); del model, encoder
    
    metrics = {
        "R2 Score": r2_score(y_test, y_pred),
        "Pearson Correlation": stats.pearsonr(y_test, y_pred)[0],
        "Spearman Correlation": stats.spearmanr(y_test, y_pred)[0],
        "Condition": condition_name
    }
    return metrics

def run_experiment(data_cond, target_name, feature_cols, condition_name, output_dir):
    X = data_cond[feature_cols].values
    y = data_cond[target_name].values.astype(float)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return _train_and_evaluate_regression_model(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
        input_dim=X_train_scaled.shape[1], condition_name=condition_name, output_dir=output_dir
    )

def run_cross_rater_experiment(train_df, test_df, target_name, feature_cols, condition_name):
    if train_df.empty or test_df.empty: return None
    X_train_full = train_df[feature_cols].values
    y_train_full = train_df[target_name].values.astype(float)
    X_test = test_df[feature_cols].values
    y_test = test_df[target_name].values.astype(float)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return _train_and_evaluate_regression_model(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
        input_dim=X_train_scaled.shape[1], condition_name=condition_name, output_dir=None
    )

def main(output_dir="results", preloaded_df=None):
    os.makedirs(output_dir, exist_ok=True)
    df = preloaded_df if preloaded_df is not None else load_and_preprocess_data(CONFIG, rater_id=CONFIG.get('rater_id'))
    if df is None or df.empty: return
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if not feature_cols: return
    df = df.dropna(subset=feature_cols)
    conditions = { 'beauty_abstract': 'Beauty', 'liking_abstract': 'Liking', 'beauty_represent': 'Beauty', 'liking_represent': 'Liking' }
    all_results = []
    for name, target in conditions.items():
        data_cond = df[(df['Representational'] == (1 if 'represent' in name else 0)) & (df[target].notna())]
        if not data_cond.empty and len(data_cond) >= 20:
            metrics = run_experiment(data_cond, target, feature_cols, name, output_dir)
            metrics['Experiment'] = "Standard_Regression"
            all_results.append(metrics)
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_dir, "regression_results.csv"), index=False)

if __name__ == '__main__':
    main()