import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Dropout, BatchNormalization
from tensorflow.keras import backend as K
import pickle

# --- CONFIGURATION ---
CONFIG = {
    # CORRECTED: This path now correctly points up one directory to your main Data folder
    "data_dir": "../Data", 
    
    # CORRECTED: The cache will now also be stored Data folder
    "cache_dir": os.path.join("../Data", "features_cache"),
    
    "objective_ratings_csv": "PaintingDataMeans.csv",
    "abstract_dir_name": "Abstract_Images",
    "representational_dir_name": "Representational_Images",
    "rating_files": {
        'Abstract_Beauty': ('Abstract_All_Raters.csv', 'Beauty', 0),
        'Abstract_Liking': ('Abstract_Liking_All_Raters.csv', 'Liking', 0),
        'Rep_Beauty': ('Representational_All_Raters.csv', 'Beauty', 1),
        'Rep_Liking': ('Representational_Liking_All_Raters.csv', 'Liking', 1),
    },
    "image_target_size": (224, 224),
    "batch_size": 32,
}
FEATURE_PREFIX = "ResNet50_feat_"

# --- Utility functions ---
def _safe_read_csv(path):
    if not os.path.isabs(path):
        path = os.path.join(CONFIG["data_dir"], path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

def _extract_first_int(s):
    if pd.isna(s): return None
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else None

def build_file_map(directory):
    mapping = {}
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    for fn in os.listdir(directory):
        key = _extract_first_int(fn)
        if key is not None:
            mapping[key] = fn
    return mapping

# --- Feature extraction with ResNet50 ---
def batch_extract_features(image_paths, batch_size, target_size):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*target_size, 3))
    pooling = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=pooling)
    features_list, valid_idx = [], []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Batch Feature Extraction", leave=False):
        batch_paths = image_paths[start:start+batch_size]
        imgs, succeeded_local_idxs = [], []
        for j, p in enumerate(batch_paths):
            try:
                img = load_img(p, target_size=target_size)
                imgs.append(img_to_array(img))
                succeeded_local_idxs.append(start + j)
            except Exception:
                continue
        if not imgs: continue
        batch_input = preprocess_input(np.array(imgs))
        feats = model.predict(batch_input, verbose=0)
        features_list.append(feats)
        valid_idx.extend(succeeded_local_idxs)
    if not features_list:
        return np.empty((0, model.output_shape[1])), []
    K.clear_session()
    del model
    return np.vstack(features_list), valid_idx

# --- Shared MLP encoder ---
def build_mlp_encoder(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    return Model(inputs=inp, outputs=x, name="mlp_encoder")

# --- Main Data Loader ---
def get_image_paths_and_ratings(config=CONFIG, rater_id=None, exclude_rater_id=None):
    df_objective = _safe_read_csv(config["objective_ratings_csv"])
    processed_dfs = {}

    for key, (file_name, rating_col, rep_code) in config["rating_files"].items():
        df_rater_raw = _safe_read_csv(file_name)
        df_rater_raw['PaintingID'] = df_rater_raw['Painting'].astype(str).str.extract(r'(\d+)')[0].astype(int)

        if 'Rater' not in df_rater_raw.columns:
            raise ValueError(f"'Rater' column not found in {file_name}. Cannot run rater-specific analysis.")
        
        if exclude_rater_id is not None:
            df_rater_raw = df_rater_raw[df_rater_raw['Rater'] != exclude_rater_id].copy()

        if rater_id is not None:
            df_rater = df_rater_raw[df_rater_raw['Rater'] == rater_id].copy()
            if df_rater.empty:
                continue
            df_processed = df_rater[['PaintingID', rating_col]].copy()
        else: # Average mode
            avg_series = df_rater_raw.groupby('PaintingID')[rating_col].mean()
            df_processed = avg_series.reset_index()

        df_processed['Painting'] = df_processed['PaintingID']
        df_processed['Representational'] = rep_code
        processed_dfs[key] = df_processed

    df_abstract = pd.merge(processed_dfs.get('Abstract_Beauty', pd.DataFrame()), processed_dfs.get('Abstract_Liking', pd.DataFrame()), on=['Painting', 'Representational', 'PaintingID'], how='outer')
    df_rep = pd.merge(processed_dfs.get('Rep_Beauty', pd.DataFrame()), processed_dfs.get('Rep_Liking', pd.DataFrame()), on=['Painting', 'Representational', 'PaintingID'], how='outer')
    df_combined = pd.concat([df_abstract, df_rep], ignore_index=True)
    df_objective['Painting'] = df_objective['Painting'].astype(str).str.extract(r'(\d+)')[0].astype(int)
    df_final = pd.merge(df_objective, df_combined, on=['Painting', 'Representational'], how='left')

    abstract_dir = os.path.join(config["data_dir"], config["abstract_dir_name"])
    represent_dir = os.path.join(config["data_dir"], config["representational_dir_name"])
    abstract_map = build_file_map(abstract_dir)
    represent_map = build_file_map(represent_dir)

    def map_path(row):
        pid = int(row['Painting'])
        fm, d = (abstract_map, abstract_dir) if row['Representational'] == 0 else (represent_map, represent_dir)
        fn = fm.get(pid)
        return os.path.join(d, fn) if fn else None

    df_final['filepath'] = df_final.apply(map_path, axis=1)
    return df_final.dropna(subset=['filepath']).reset_index(drop=True)

def load_and_preprocess_data(config=CONFIG, rater_id=None, exclude_rater_id=None, use_cache=True):
    os.makedirs(config['cache_dir'], exist_ok=True)
    
    rater_str = f"rater_{rater_id}" if rater_id else "rater_avg"
    exclude_str = f"exclude_{exclude_rater_id}" if exclude_rater_id else "exclude_none"
    cache_filename = f"{rater_str}_{exclude_str}.pkl"
    cache_path = os.path.join(config['cache_dir'], cache_filename)
    
    if use_cache and os.path.exists(cache_path):
        print(f"--> Loading features from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("--> No cache found. Processing data and extracting features...")
    df = get_image_paths_and_ratings(config, rater_id=rater_id, exclude_rater_id=exclude_rater_id)
    if df.empty: return pd.DataFrame()
    
    features_array, valid_idx = batch_extract_features(df['filepath'].tolist(), config["batch_size"], config["image_target_size"])
    if features_array.size == 0: return df.iloc[valid_idx]
    
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    feat_dim = features_array.shape[1]
    feature_cols = [f"{FEATURE_PREFIX}{i}" for i in range(feat_dim)]
    features_df = pd.DataFrame(features_array, columns=feature_cols)
    
    final_df = pd.concat([df_valid, features_df], axis=1)
    
    if use_cache:
        print(f"--> Saving features to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(final_df, f)
            
    return final_df