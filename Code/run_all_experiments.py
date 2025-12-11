import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
from sklearn.model_selection import train_test_split

# --- Main Imports ---
import data_preprocessor_resnet50 as data_loader
from data_preprocessor_resnet50 import build_mlp_encoder, FEATURE_PREFIX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Subtract, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

# --- CONFIGURATION ---
RESULT_DIR = "Final_results"  # Same directory structure as first approach
NUM_RUNS = 10
NUM_RATERS = 5  # Adjust based on your dataset
PAIRS_PER_IMAGE_VALUES = range(1, 11)  # N values from 1 to 15
NUM_WORKERS = max(1, os.cpu_count() - 1)

# =================================================================================
# ===== REGRESSION MODEL FUNCTIONS =====
# =================================================================================

def build_regression_model_from_encoder(encoder):
    """Build regression model from encoder"""
    output = Dense(1, activation='linear', name='reg_output')(encoder.output)
    return Model(inputs=encoder.input, outputs=output)

def run_regression_single(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    """Train and evaluate regression model for a single run"""
    encoder = build_mlp_encoder(input_dim)
    model = build_regression_model_from_encoder(encoder)
    
    optimizer = SGD(learning_rate=0.005, momentum=0.9)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=0)
    
    model.fit(X_train, y_train, epochs=300, batch_size=32,
              validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
    
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = r2_score(y_test, y_pred)
    rho = pearsonr(y_test, y_pred)[0]
    rs = spearmanr(y_test, y_pred)[0]
    
    K.clear_session()
    del model, encoder
    
    return {"mae": mae, "r2": r2, "rho": rho, "rs": rs}

# =================================================================================
# ===== PAIRWISE MODEL FUNCTIONS =====
# =================================================================================

def make_pairs_from_rows(rows_df, target_col, pairs_per_image, seed=None):
    """Generate pairwise comparisons from rows"""
    rng = np.random.default_rng(seed)
    rows = rows_df.reset_index(drop=True)
    pairs = []
    n_rows = len(rows)
    if n_rows <= 1:
        return pd.DataFrame()
    
    all_opponents = {i: rng.permutation([j for j in range(n_rows) if j != i]) for i in range(n_rows)}
    
    for i, r1 in rows.iterrows():
        chosen_opponents = all_opponents[i][:pairs_per_image]
        for j in chosen_opponents:
            preference = 1 if r1[target_col] > rows.loc[j, target_col] else -1
            pairs.append({'idx1': int(i), 'idx2': int(j), 'preference': int(preference)})
    
    return pd.DataFrame(pairs)

def build_siamese_on_features(input_dim):
    """Build siamese network for pairwise learning"""
    encoder = build_mlp_encoder(input_dim)
    input_A, input_B = Input(shape=(input_dim,)), Input(shape=(input_dim,))
    encoded_A, encoded_B = encoder(input_A), encoder(input_B)
    diff = Subtract()([encoded_A, encoded_B])
    output = Dense(1, activation='tanh', name='preference')(diff)
    return Model(inputs=[input_A, input_B], outputs=output)

def build_inputs_from_pairs(pair_df, features_matrix):
    """Build input arrays from pair dataframe"""
    A = features_matrix[pair_df['idx1'].values]
    B = features_matrix[pair_df['idx2'].values]
    y = pair_df['preference'].values.astype(np.float32)
    return [A, B], y

def run_pairwise_single(train_rows, train_pairs, val_rows, val_pairs, test_rows, test_pairs, 
                       feature_cols, target_col, pairs_per_image):
    """Train and evaluate pairwise model for a single run"""
    X_train_feats = train_rows[feature_cols].values
    X_val_feats = val_rows[feature_cols].values
    X_test_feats = test_rows[feature_cols].values
    
    train_X, train_y = build_inputs_from_pairs(train_pairs, X_train_feats)
    val_X, val_y = build_inputs_from_pairs(val_pairs, X_val_feats)
    test_X, test_y = build_inputs_from_pairs(test_pairs, X_test_feats)
    
    model = build_siamese_on_features(X_train_feats.shape[1])
    model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.9), loss='hinge', metrics=['mae'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    model.fit(train_X, train_y, epochs=100, batch_size=32, 
              validation_data=(val_X, val_y), callbacks=[early_stopping], verbose=0)
    
    y_pred_proba = model.predict(test_X, verbose=0).flatten()
    y_true = test_y.astype(float)
    
    # Calculate correlations on preference predictions
    rho = pearsonr(y_true, y_pred_proba)[0]
    rs = spearmanr(y_true, y_pred_proba)[0]
    
    K.clear_session()
    del model
    
    return {"rho": rho, "rs": rs}

# =================================================================================
# ===== WORKER FUNCTIONS FOR MULTIPROCESSING =====
# =================================================================================

def regression_worker(args):
    """Worker for regression experiments"""
    run_id, X, y, feature_cols, seed = args
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    result = run_regression_single(X_train_scaled, y_train, X_val_scaled, y_val, 
                                   X_test_scaled, y_test, X_train_scaled.shape[1])
    result['Run'] = run_id
    return result

def pairwise_worker(args):
    """Worker for pairwise experiments"""
    run_id, df, target_col, feature_cols, pairs_per_image, seed = args
    
    # Split paintings
    train_val, test = train_test_split(df, test_size=0.2, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.25, random_state=seed)
    
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    # Create pairs
    train_pairs = make_pairs_from_rows(train, target_col, pairs_per_image, seed)
    val_pairs = make_pairs_from_rows(val, target_col, pairs_per_image, seed)
    test_pairs = make_pairs_from_rows(test, target_col, pairs_per_image, seed)
    
    if train_pairs.empty or test_pairs.empty:
        return None
    
    result = run_pairwise_single(train, train_pairs, val, val_pairs, test, test_pairs,
                                feature_cols, target_col, pairs_per_image)
    result['Run'] = run_id
    result['N'] = pairs_per_image
    return result

# =================================================================================
# ===== EXPERIMENT ORCHESTRATION FUNCTIONS =====
# =================================================================================

def run_average_experiments(df, painting_type, rating_type):
    """Run average rating experiments (regression + pairwise)"""
    print(f"\n{'='*70}")
    print(f"Running Average Experiments: {painting_type} - {rating_type}")
    print(f"{'='*70}")
    
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    target_col = 'Beauty' if rating_type == 'beauty' else 'Liking'
    
    # Filter data
    rep_code = 1 if painting_type == 'representational' else 0
    df_filtered = df[(df['Representational'] == rep_code) & (df[target_col].notna())].reset_index(drop=True)
    
    if len(df_filtered) < 20:
        print(f"Insufficient data for {painting_type} - {rating_type}")
        return
    
    X = df_filtered[feature_cols].values
    y = df_filtered[target_col].values.astype(float)
    
    # ===== REGRESSION =====
    print(f"\nRunning regression experiments (10 runs)...")
    reg_tasks = [(i, X, y, feature_cols, i) for i in range(NUM_RUNS)]
    
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        reg_results = list(tqdm(pool.imap(regression_worker, reg_tasks), total=len(reg_tasks), desc="Regression"))
    
    # Average and save
    reg_df = pd.DataFrame(reg_results)
    reg_avg = reg_df[['mae', 'r2', 'rho', 'rs']].mean().to_dict()
    
    output_file = os.path.join(RESULT_DIR, f"{painting_type}_{rating_type}_average.csv")
    pd.DataFrame([reg_avg]).to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    
    # ===== PAIRWISE =====
    print(f"\nRunning pairwise experiments (10 runs × 15 N values)...")
    pair_tasks = [(i, df_filtered, target_col, feature_cols, n, i) 
                  for n in PAIRS_PER_IMAGE_VALUES 
                  for i in range(NUM_RUNS)]
    
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        pair_results = [r for r in tqdm(pool.imap(pairwise_worker, pair_tasks), 
                                       total=len(pair_tasks), desc="Pairwise") if r is not None]
    
    # Average by N and save
    pair_df = pd.DataFrame(pair_results)
    pair_avg = pair_df.groupby('N')[['rho', 'rs']].mean().reset_index()
    
    output_file = os.path.join(RESULT_DIR, f"{painting_type}_{rating_type}_average_comparative.csv")
    pair_avg.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")

def run_within_rater_experiments(painting_type, rating_type):
    """Run within-rater experiments (regression + pairwise) for each rater"""
    print(f"\n{'='*70}")
    print(f"Running Within-Rater Experiments: {painting_type} - {rating_type}")
    print(f"{'='*70}")
    
    all_reg_results = []
    all_pair_results = []
    
    for rater in range(1, NUM_RATERS + 1):
        print(f"\n--- Processing Rater {rater}/{NUM_RATERS} ---")
        
        # Load rater-specific data
        df = data_loader.load_and_preprocess_data(rater_id=rater)
        if df.empty:
            print(f"No data for rater {rater}, skipping...")
            continue
        
        feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
        target_col = 'Beauty' if rating_type == 'beauty' else 'Liking'
        
        # Filter data
        rep_code = 1 if painting_type == 'representational' else 0
        df_filtered = df[(df['Representational'] == rep_code) & (df[target_col].notna())].reset_index(drop=True)
        
        if len(df_filtered) < 20:
            print(f"Insufficient data for rater {rater}")
            continue
        
        X = df_filtered[feature_cols].values
        y = df_filtered[target_col].values.astype(float)
        
        # ===== REGRESSION =====
        print(f"  Running regression (10 runs)...")
        reg_tasks = [(i, X, y, feature_cols, i + rater*1000) for i in range(NUM_RUNS)]
        
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            reg_results = list(pool.imap(regression_worker, reg_tasks))
        
        # Average for this rater
        reg_df = pd.DataFrame(reg_results)
        reg_avg = reg_df[['mae', 'r2', 'rho', 'rs']].mean().to_dict()
        reg_avg['Rater'] = rater
        all_reg_results.append(reg_avg)
        
        # ===== PAIRWISE =====
        print(f"  Running pairwise (10 runs × 15 N values)...")
        pair_tasks = [(i, df_filtered, target_col, feature_cols, n, i + rater*1000) 
                      for n in PAIRS_PER_IMAGE_VALUES 
                      for i in range(NUM_RUNS)]
        
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            pair_results = [r for r in pool.imap(pairwise_worker, pair_tasks) if r is not None]
        
        # Average by N for this rater
        pair_df = pd.DataFrame(pair_results)
        pair_avg = pair_df.groupby('N')[['rho', 'rs']].mean().reset_index()
        pair_avg['Rater'] = rater
        all_pair_results.append(pair_avg)
    
    # Save all raters
    if all_reg_results:
        reg_output = pd.DataFrame(all_reg_results)
        reg_output = reg_output[['Rater', 'mae', 'r2', 'rho', 'rs']]
        output_file = os.path.join(RESULT_DIR, f"{painting_type}_{rating_type}_within.csv")
        reg_output.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file}")
    
    if all_pair_results:
        pair_output = pd.concat(all_pair_results, ignore_index=True)
        pair_output = pair_output[['Rater', 'N', 'rho', 'rs']]
        output_file = os.path.join(RESULT_DIR, f"{painting_type}_{rating_type}_within_comparative.csv")
        pair_output.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")

def run_cross_rater_experiments(painting_type, rating_type):
    """Run cross-rater experiments (regression + pairwise) for each rater"""
    print(f"\n{'='*70}")
    print(f"Running Cross-Rater Experiments: {painting_type} - {rating_type}")
    print(f"{'='*70}")
    
    all_reg_results = []
    all_pair_results = []
    
    for rater in range(1, NUM_RATERS + 1):
        print(f"\n--- Processing Rater {rater}/{NUM_RATERS} (Train on others, test on this rater) ---")
        
        # Load training data (exclude this rater)
        train_df = data_loader.load_and_preprocess_data(exclude_rater_id=rater)
        # Load test data (only this rater)
        test_df = data_loader.load_and_preprocess_data(rater_id=rater)
        
        if train_df.empty or test_df.empty:
            print(f"Insufficient data for rater {rater}, skipping...")
            continue
        
        feature_cols = [c for c in train_df.columns if c.startswith(FEATURE_PREFIX)]
        target_col = 'Beauty' if rating_type == 'beauty' else 'Liking'
        
        # Filter data
        rep_code = 1 if painting_type == 'representational' else 0
        train_filtered = train_df[(train_df['Representational'] == rep_code) & (train_df[target_col].notna())].reset_index(drop=True)
        test_filtered = test_df[(test_df['Representational'] == rep_code) & (test_df[target_col].notna())].reset_index(drop=True)
        
        if len(train_filtered) < 20 or len(test_filtered) < 10:
            print(f"Insufficient data for rater {rater}")
            continue
        
        # ===== REGRESSION =====
        print(f"  Running regression (10 runs)...")
        reg_results_rater = []
        
        for run_id in range(NUM_RUNS):
            # Split train into train/val
            X_train_full = train_filtered[feature_cols].values
            y_train_full = train_filtered[target_col].values.astype(float)
            X_test = test_filtered[feature_cols].values
            y_test = test_filtered[target_col].values.astype(float)
            
            X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, 
                                                               test_size=0.2, random_state=run_id + rater*1000)
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            result = run_regression_single(X_train_scaled, y_train, X_val_scaled, y_val,
                                         X_test_scaled, y_test, X_train_scaled.shape[1])
            reg_results_rater.append(result)
        
        # Average for this rater
        reg_df = pd.DataFrame(reg_results_rater)
        reg_avg = reg_df[['mae', 'r2', 'rho', 'rs']].mean().to_dict()
        reg_avg['Rater'] = rater
        all_reg_results.append(reg_avg)
        
        # ===== PAIRWISE =====
        print(f"  Running pairwise (10 runs × 15 N values)...")
        pair_results_rater = []
        
        for n in PAIRS_PER_IMAGE_VALUES:
            for run_id in range(NUM_RUNS):
                # Split train into train/val
                train_temp, _ = train_test_split(train_filtered, test_size=0.2, random_state=run_id + rater*1000)
                train_rows, val_rows = train_test_split(train_temp, test_size=0.25, random_state=run_id + rater*1000)
                
                train_rows = train_rows.reset_index(drop=True)
                val_rows = val_rows.reset_index(drop=True)
                test_rows = test_filtered.reset_index(drop=True)
                
                # Create pairs
                train_pairs = make_pairs_from_rows(train_rows, target_col, n, run_id + rater*1000)
                val_pairs = make_pairs_from_rows(val_rows, target_col, n, run_id + rater*1000)
                test_pairs = make_pairs_from_rows(test_rows, target_col, n, run_id + rater*1000)
                
                if train_pairs.empty or test_pairs.empty:
                    continue
                
                result = run_pairwise_single(train_rows, train_pairs, val_rows, val_pairs, 
                                           test_rows, test_pairs, feature_cols, target_col, n)
                result['N'] = n
                pair_results_rater.append(result)
        
        # Average by N for this rater
        if pair_results_rater:
            pair_df = pd.DataFrame(pair_results_rater)
            pair_avg = pair_df.groupby('N')[['rho', 'rs']].mean().reset_index()
            pair_avg['Rater'] = rater
            all_pair_results.append(pair_avg)
    
    # Save all raters
    if all_reg_results:
        reg_output = pd.DataFrame(all_reg_results)
        reg_output = reg_output[['Rater', 'mae', 'r2', 'rho', 'rs']]
        output_file = os.path.join(RESULT_DIR, f"{painting_type}_{rating_type}_cross.csv")
        reg_output.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file}")
    
    if all_pair_results:
        pair_output = pd.concat(all_pair_results, ignore_index=True)
        pair_output = pair_output[['Rater', 'N', 'rho', 'rs']]
        output_file = os.path.join(RESULT_DIR, f"{painting_type}_{rating_type}_cross_comparative.csv")
        pair_output.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")

# =================================================================================
# ===== MAIN EXECUTION =====
# =================================================================================

def main():
    """Main execution function"""
    # Configuration
    PAINTING_TYPES = ["representational"]
    RATING_TYPES = ["liking"]
    
    # Choose which experiments to run
    RUN_AVERAGE = False
    RUN_WITHIN = False
    RUN_CROSS = True
    
    # Create result directory
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    print("\n" + "="*70)
    print("STARTING EXPERIMENTS")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Result Directory: {RESULT_DIR}")
    print(f"  - Number of Runs: {NUM_RUNS}")
    print(f"  - Number of Workers: {NUM_WORKERS}")
    print(f"  - Experiments: Average={RUN_AVERAGE}, Within={RUN_WITHIN}, Cross={RUN_CROSS}")
    print("="*70)
    
    # ===== AVERAGE EXPERIMENTS =====
    if RUN_AVERAGE:
        print("\n" + "="*70)
        print("AVERAGE RATING EXPERIMENTS")
        print("="*70)
        
        # Load average data once
        avg_df = data_loader.load_and_preprocess_data()
        
        for painting in PAINTING_TYPES:
            for rating in RATING_TYPES:
                run_average_experiments(avg_df, painting, rating)
    
    # ===== WITHIN-RATER EXPERIMENTS =====
    if RUN_WITHIN:
        print("\n" + "="*70)
        print("WITHIN-RATER EXPERIMENTS")
        print("="*70)
        
        for painting in PAINTING_TYPES:
            for rating in RATING_TYPES:
                run_within_rater_experiments(painting, rating)
    
    # ===== CROSS-RATER EXPERIMENTS =====
    if RUN_CROSS:
        print("\n" + "="*70)
        print("CROSS-RATER EXPERIMENTS")
        print("="*70)
        
        for painting in PAINTING_TYPES:
            for rating in RATING_TYPES:
                run_cross_rater_experiments(painting, rating)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nResults saved in: {RESULT_DIR}/")
    print("You can now run plot_results.py to generate visualizations.")

if __name__ == '__main__':
    main()