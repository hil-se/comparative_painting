import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Subtract, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

from data_preprocessor_resnet50 import (load_and_preprocess_data, CONFIG, 
                                        build_mlp_encoder, FEATURE_PREFIX)

def make_pairs_from_rows(rows_df, target_col, pairs_per_image, seed=None):
    rng = np.random.default_rng(seed)
    rows = rows_df.reset_index(drop=True)
    pairs = []
    n_rows = len(rows)
    if n_rows <= 1: return pd.DataFrame()
    all_opponents = {i: rng.permutation([j for j in range(n_rows) if j != i]) for i in range(n_rows)}
    for i, r1 in rows.iterrows():
        chosen_opponents = all_opponents[i][:pairs_per_image]
        for j in chosen_opponents:
            preference = 1 if r1[target_col] > rows.loc[j, target_col] else -1
            pairs.append({'idx1': int(i), 'idx2': int(j), 'preference': int(preference)})
    return pd.DataFrame(pairs)

def split_paintings_and_create_pairs(df, target_col, pairs_per_image, seed=None):
    train_val, test = train_test_split(df, test_size=0.2, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.25, random_state=seed)
    return {
        'train': (train.reset_index(drop=True), make_pairs_from_rows(train, target_col, pairs_per_image, seed)),
        'val': (val.reset_index(drop=True), make_pairs_from_rows(val, target_col, pairs_per_image, seed)),
        'test': (test.reset_index(drop=True), make_pairs_from_rows(test, target_col, pairs_per_image, seed))
    }

def build_siamese_on_features(input_dim):
    encoder = build_mlp_encoder(input_dim)
    input_A, input_B = Input(shape=(input_dim,)), Input(shape=(input_dim,))
    encoded_A, encoded_B = encoder(input_A), encoder(input_B)
    diff = Subtract()([encoded_A, encoded_B])
    output = Dense(1, activation='tanh', name='preference')(diff)
    return Model(inputs=[input_A, input_B], outputs=output)

def build_inputs_from_pairs(pair_df, features_matrix):
    A = features_matrix[pair_df['idx1'].values]
    B = features_matrix[pair_df['idx2'].values]
    y = pair_df['preference'].values.astype(np.float32)
    return [A, B], y

def _train_and_evaluate_pairwise_model(train_X, train_y, val_X, val_y, test_X, test_y, input_dim, condition_name):
    model = build_siamese_on_features(input_dim)
    model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.9), loss='hinge', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    fit_args = {'x': train_X, 'y': train_y, 'epochs': 100, 'batch_size': CONFIG["batch_size"], 'callbacks': [early_stopping], 'verbose': 0}
    if val_X is not None and val_y is not None:
        fit_args['validation_data'] = (val_X, val_y)
    model.fit(**fit_args)
    y_pred_proba = model.predict(test_X).flatten()
    y_true, y_true_bin = test_y.astype(int), (test_y == 1).astype(int)
    y_scores = (y_pred_proba + 1.0) / 2.0
    K.clear_session(); del model
    metrics = {
        "Accuracy": accuracy_score(y_true, np.sign(y_pred_proba)),
        "ROC_AUC": roc_auc_score(y_true_bin, y_scores) if len(np.unique(y_true_bin)) > 1 else np.nan,
        "Pearson_Corr": pearsonr(y_true.astype(float), y_pred_proba)[0],
        "Spearman_Corr": spearmanr(y_true.astype(float), y_pred_proba)[0],
        "Condition": condition_name
    }
    return metrics

def run_experiment_on_condition(df_cond, target_name, pairs_per_image, condition_name):
    splits = split_paintings_and_create_pairs(df_cond, target_name, pairs_per_image)
    (train_rows, train_pairs), (val_rows, val_pairs), (test_rows, test_pairs) = splits['train'], splits['val'], splits['test']
    if train_pairs.empty or test_pairs.empty: return None
    feat_cols = [c for c in df_cond.columns if c.startswith(FEATURE_PREFIX)]
    X_train_feats, X_val_feats, X_test_feats = train_rows[feat_cols].values, val_rows[feat_cols].values, test_rows[feat_cols].values
    train_X, train_y = build_inputs_from_pairs(train_pairs, X_train_feats)
    val_X, val_y = build_inputs_from_pairs(val_pairs, X_val_feats) if not val_pairs.empty else (None, None)
    test_X, test_y = build_inputs_from_pairs(test_pairs, X_test_feats)
    return _train_and_evaluate_pairwise_model(
        train_X, train_y, val_X, val_y, test_X, test_y,
        input_dim=X_train_feats.shape[1], condition_name=condition_name
    )

def run_cross_rater_pairwise_experiment(train_df, test_df, target_name, pairs_per_image, feature_cols, condition_name):
    if train_df.empty or test_df.empty or len(test_df) <= pairs_per_image: return None
    train_val, _ = train_test_split(train_df, test_size=0.2)
    train_rows, val_rows = train_test_split(train_val, test_size=0.25)
    train_pairs = make_pairs_from_rows(train_rows, target_name, pairs_per_image)
    val_pairs = make_pairs_from_rows(val_rows, target_name, pairs_per_image)
    test_pairs = make_pairs_from_rows(test_df.reset_index(drop=True), target_name, pairs_per_image)
    if train_pairs.empty or test_pairs.empty: return None
    X_train_feats, X_val_feats, X_test_feats = train_rows[feature_cols].values, val_rows[feature_cols].values, test_df[feature_cols].values
    train_X, train_y = build_inputs_from_pairs(train_pairs, X_train_feats)
    val_X, val_y = build_inputs_from_pairs(val_pairs, X_val_feats)
    test_X, test_y = build_inputs_from_pairs(test_pairs, X_test_feats)
    metrics = _train_and_evaluate_pairwise_model(
        train_X, train_y, val_X, val_y, test_X, test_y,
        input_dim=X_train_feats.shape[1], condition_name=condition_name
    )
    if metrics:
        metrics['n'] = pairs_per_image
    return metrics

def main(output_dir="results", specific_n=None, preloaded_df=None):
    os.makedirs(output_dir, exist_ok=True)
    full_df = preloaded_df if preloaded_df is not None else load_and_preprocess_data(CONFIG, rater_id=CONFIG.get('rater_id'))
    if full_df is None or full_df.empty: return
    feature_cols = [c for c in full_df.columns if c.startswith(FEATURE_PREFIX)]
    full_df = full_df.dropna(subset=feature_cols).reset_index(drop=True)
    conditions = { 'beauty_abstract': 'Beauty', 'liking_abstract': 'Liking', 'beauty_represent': 'Beauty', 'liking_represent': 'Liking' }
    n_values = [specific_n] if specific_n is not None else range(1, 16)
    all_run_results = []
    for n_pairs in n_values:
        for name, target in conditions.items():
            data_cond = full_df[(full_df['Representational'] == (1 if 'represent' in name else 0)) & (full_df[target].notna())]
            if len(data_cond) > n_pairs and len(data_cond) >= 20:
                metrics = run_experiment_on_condition(data_cond, target, n_pairs, name)
                if metrics:
                    metrics['Experiment'] = f"Standard_Pairwise_n{n_pairs}"
                    all_run_results.append(metrics)
    if all_run_results:
        pd.DataFrame(all_run_results).to_csv(os.path.join(output_dir, "pairwise_results.csv"), index=False)

if __name__ == '__main__':
    main()