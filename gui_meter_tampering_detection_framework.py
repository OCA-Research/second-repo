# filename: meter_tampering_detection_framework.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

# Deep Learning specific imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns
import os # To check for file existence

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
DATA_FILENAME = 'simulated_disco_data.csv'

def run_detection_framework():
    """
    Runs the meter tampering detection framework and returns evaluation results.
    """
    results = {}

    # --- 1. Load Data from CSV ---
    print(f"\n--- 1. Loading data from '{DATA_FILENAME}' ---")
    if not os.path.exists(DATA_FILENAME):
        return {"error": f"Error: Data file '{DATA_FILENAME}' not found. Please run 'generate_simulated_data.py' first to create the data file."}

    df_raw = pd.read_csv(DATA_FILENAME)
    results['initial_data_shape'] = df_raw.shape
    results['actual_tampering_rate'] = df_raw.drop_duplicates(subset=['customer_id'])['is_tampering_customer'].mean()


    # --- 2. Advanced Data Preprocessing and Feature Engineering (for XGBoost) ---
    print("\n--- 2. Advanced Data Preprocessing & Feature Engineering (for XGBoost) ---")

    df_xgb = df_raw.copy() # Separate dataframe for XGBoost (monthly level processing)

    # Fill missing values
    for col in ['consumption_kwh', 'billed_amount_ngn']:
        df_xgb[col] = df_xgb[col].fillna(df_xgb.groupby('customer_id')[col].transform('mean'))
        df_xgb[col] = df_xgb[col].fillna(0) # Fill any remaining NaNs (e.g., new customers with no history)


    # Feature Engineering
    df_xgb['billing_to_consumption_ratio'] = df_xgb['billed_amount_ngn'] / (df_xgb['consumption_kwh'] + 1e-6)
    df_xgb['billing_to_consumption_ratio'] = df_xgb['billing_to_consumption_ratio'].replace([np.inf, -np.inf], 0)

    df_xgb['consumption_last_month'] = df_xgb.groupby('customer_id')['consumption_kwh'].shift(1).fillna(0)
    df_xgb['consumption_diff'] = df_xgb['consumption_kwh'] - df_xgb['consumption_last_month']
    df_xgb['consumption_ratio_last_month'] = df_xgb['consumption_kwh'] / (df_xgb['consumption_last_month'] + 1e-6)
    df_xgb['consumption_ratio_last_month'] = df_xgb['consumption_ratio_last_month'].replace([np.inf, -np.inf], 0)

    df_xgb['consumption_3m_avg'] = df_xgb.groupby('customer_id')['consumption_kwh'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df_xgb['consumption_deviation_from_3m_avg'] = df_xgb['consumption_kwh'] - df_xgb['consumption_3m_avg']

    df_xgb['customer_avg_consumption'] = df_xgb.groupby('customer_id')['consumption_kwh'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    df_xgb['consumption_deviation_from_avg'] = df_xgb['consumption_kwh'] - df_xgb['customer_avg_consumption']
    df_xgb['consumption_x_deviation'] = df_xgb['consumption_kwh'] * df_xgb['consumption_deviation_from_avg']

    # One-Hot Encode categorical features
    df_xgb = pd.get_dummies(df_xgb, columns=['payment_history', 'customer_category'], drop_first=True)

    # Define features (X_xgb) and target (y_xgb) for XGBoost
    X_xgb = df_xgb.drop(columns=['customer_id', 'month', 'is_tampering_month', 'is_tampering_customer'])
    y_xgb = df_xgb['is_tampering_month']


    # --- 3. Deep Learning Data Preparation (for Autoencoder) ---
    print("\n--- 3. Deep Learning Data Preparation (for Autoencoder) ---")

    df_dl = df_raw.copy() # Separate dataframe for Deep Learning (customer sequence level)

    # Fill missing values for DL data
    df_dl['consumption_kwh'] = df_dl['consumption_kwh'].fillna(0)
    df_dl['billed_amount_ngn'] = df_dl['billed_amount_ngn'].fillna(0)

    # Feature Engineering for DL
    df_dl['billing_to_consumption_ratio'] = df_dl['billed_amount_ngn'] / (df_dl['consumption_kwh'] + 1e-6)
    df_dl['billing_to_consumption_ratio'] = df_dl['billing_to_consumption_ratio'].replace([np.inf, -np.inf], 0)

    df_dl['consumption_last_month'] = df_dl.groupby('customer_id')['consumption_kwh'].shift(1).fillna(0)
    df_dl['consumption_diff'] = df_dl['consumption_kwh'] - df_dl['consumption_last_month']
    df_dl['consumption_ratio_last_month'] = df_dl['consumption_kwh'] / (df_dl['consumption_last_month'] + 1e-6)
    df_dl['consumption_ratio_last_month'] = df_dl['consumption_ratio_last_month'].replace([np.inf, -np.inf], 0)

    df_dl['consumption_3m_avg'] = df_dl.groupby('customer_id')['consumption_kwh'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df_dl['consumption_deviation_from_3m_avg'] = df_dl['consumption_kwh'] - df_dl['consumption_3m_avg']

    df_dl['customer_avg_consumption'] = df_dl.groupby('customer_id')['consumption_kwh'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    df_dl['consumption_deviation_from_avg'] = df_dl['consumption_kwh'] - df_dl['customer_avg_consumption']
    df_dl['consumption_x_deviation'] = df_dl['consumption_kwh'] * df_dl['consumption_deviation_from_avg']

    df_dl = pd.get_dummies(df_dl, columns=['payment_history', 'customer_category'], drop_first=True)

    feature_cols_dl = [col for col in df_dl.columns if col not in ['customer_id', 'month', 'is_tampering_month', 'is_tampering_customer']]

    # Reshape Data for Deep Learning (Sequence Format)
    num_months = df_raw['month'].max()
    grouped_dl = df_dl.groupby('customer_id')

    X_sequences_dl = []
    y_customer_labels_dl = []
    for name, group in grouped_dl:
        customer_data = group[feature_cols_dl].values
        if customer_data.shape[0] < num_months:
            padding = np.zeros((num_months - customer_data.shape[0], customer_data.shape[1]))
            customer_data = np.vstack((customer_data, padding))
        elif customer_data.shape[0] > num_months:
            customer_data = customer_data[:num_months, :]

        X_sequences_dl.append(customer_data)
        y_customer_labels_dl.append(group['is_tampering_customer'].iloc[0])

    X_sequences_dl = np.array(X_sequences_dl)
    y_customer_labels_dl = np.array(y_customer_labels_dl)

    # Scaling the 3D data (for Autoencoder)
    original_shape_dl = X_sequences_dl.shape
    X_flat_dl = X_sequences_dl.reshape(-1, original_shape_dl[2])

    scaler_dl = StandardScaler()
    X_scaled_flat_dl = scaler_dl.fit_transform(X_flat_dl)
    X_scaled_sequences_dl = X_scaled_flat_dl.reshape(original_shape_dl)


    # --- 4. Unified Data Splitting (CRITICAL for consistent evaluation) ---
    print("\n--- 4. Unified Data Splitting (CRITICAL for consistent evaluation) ---")

    customer_ids_all = df_raw['customer_id'].unique()
    y_customer_true = df_raw.drop_duplicates('customer_id')['is_tampering_customer'].values

    # Split customer IDs, not the full dataset
    train_cust_ids, test_cust_ids, _, _ = train_test_split(
        customer_ids_all, y_customer_true,
        test_size=0.3,
        random_state=42,
        stratify=y_customer_true
    )

    # Create train/test sets for XGBoost (monthly level)
    train_df_xgb = df_xgb[df_xgb['customer_id'].isin(train_cust_ids)]
    test_df_xgb = df_xgb[df_xgb['customer_id'].isin(test_cust_ids)]

    X_train_xgb = train_df_xgb.drop(columns=['customer_id', 'month', 'is_tampering_month', 'is_tampering_customer'])
    y_train_xgb = train_df_xgb['is_tampering_month']
    X_test_xgb = test_df_xgb.drop(columns=['customer_id', 'month', 'is_tampering_month', 'is_tampering_customer'])
    y_test_xgb_monthly = test_df_xgb['is_tampering_month'] # Monthly target for XGBoost's evaluation
    test_customer_ids_xgb = test_df_xgb['customer_id'] # Keep customer IDs for aggregation

    # Create train/test sets for Deep Learning (customer sequence level)
    train_indices_dl = np.where(np.isin(df_raw['customer_id'].unique(), train_cust_ids))[0]
    test_indices_dl = np.where(np.isin(df_raw['customer_id'].unique(), test_cust_ids))[0]

    X_train_dl_seq = X_scaled_sequences_dl[train_indices_dl]
    y_train_dl_cust = y_customer_labels_dl[train_indices_dl]
    X_test_dl_seq = X_scaled_sequences_dl[test_indices_dl]
    y_test_dl_cust = y_customer_labels_dl[test_indices_dl]

    # The true customer labels for the combined model will be y_test_dl_cust
    # as it represents the customer-level tampering label based on the unified split.
    y_test_combined = y_test_dl_cust # This is the ultimate ground truth for combined model

    results['num_test_customers'] = len(test_cust_ids)
    results['actual_tampering_rate_test'] = y_test_dl_cust.mean()

    # --- 5. Train XGBoost Model ---
    print("\n--- 5. Training XGBoost Model ---")
    xgb_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
        ('xgb', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                 use_label_encoder=False, random_state=42,
                                 n_estimators=100, learning_rate=0.1,
                                 max_depth=5, colsample_bytree=0.8, subsample=0.8))
    ])

    xgb_pipeline.fit(X_train_xgb, y_train_xgb)
    print("XGBoost model training complete.")

    # Get monthly predictions from XGBoost for test set
    y_pred_proba_xgb_monthly = xgb_pipeline.predict_proba(X_test_xgb)[:, 1]

    # Aggregate XGBoost predictions to customer level
    temp_df_pred_xgb = pd.DataFrame({
        'customer_id': test_customer_ids_xgb,
        'xgb_proba': y_pred_proba_xgb_monthly
    })
    y_pred_xgb_customer_level = temp_df_pred_xgb.groupby('customer_id')['xgb_proba'].max().values
    y_pred_xgb_binary_customer_level = (y_pred_xgb_customer_level > 0.5).astype(int)

    # --- 6. Train Autoencoder Model ---
    print("\n--- 6. Training Autoencoder Model ---")

    # Autoencoder trains on normal data only
    X_train_normal_dl_seq = X_train_dl_seq[y_train_dl_cust == 0]

    input_dim_ae = X_train_normal_dl_seq.shape[2]
    timesteps_ae = X_train_normal_dl_seq.shape[1]

    input_layer_ae = Input(shape=(timesteps_ae, input_dim_ae))
    encoded_ae = LSTM(64, activation='relu', return_sequences=True)(input_layer_ae)
    encoded_ae = LSTM(32, activation='relu', return_sequences=False)(encoded_ae)
    decoded_ae = RepeatVector(timesteps_ae)(encoded_ae)
    decoded_ae = LSTM(32, activation='relu', return_sequences=True)(decoded_ae)
    decoded_ae = LSTM(64, activation='relu', return_sequences=True)(decoded_ae)
    output_layer_ae = TimeDistributed(Dense(input_dim_ae))(decoded_ae)

    model_ae = Model(inputs=input_layer_ae, outputs=output_layer_ae)
    model_ae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    early_stopping_ae = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Training Autoencoder model on NORMAL data...")
    history_ae = model_ae.fit(
        X_train_normal_dl_seq, X_train_normal_dl_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping_ae],
        verbose=0
    )
    print("Autoencoder model training complete.")

    # Get reconstruction errors for all test data
    X_test_pred_ae = model_ae.predict(X_test_dl_seq, verbose=0)
    reconstruction_errors_ae = np.mean(np.square(X_test_dl_seq - X_test_pred_ae), axis=(1, 2))

    # Determine anomaly threshold for Autoencoder based on normal training data
    train_normal_pred_ae = model_ae.predict(X_train_normal_dl_seq, verbose=0)
    train_normal_errors_ae = np.mean(np.square(X_train_normal_dl_seq - train_normal_pred_ae), axis=(1, 2))
    threshold_ae = np.percentile(train_normal_errors_ae, 95)
    results['ae_anomaly_threshold'] = threshold_ae

    # Get binary predictions from Autoencoder for test set
    y_pred_ae_binary_customer_level = (reconstruction_errors_ae > threshold_ae).astype(int)

    # --- 7. Integrate Predictions ---
    print("\n--- 7. Integrating Predictions ---")

    # Strategy: Simple OR logic (flag if *either* model predicts tampering)
    combined_predictions_or = (y_pred_xgb_binary_customer_level | y_pred_ae_binary_customer_level).astype(int)


    # --- 8. Evaluate the Combined System ---
    print("\n--- 8. Evaluating the Combined System (OR Logic) ---")

    cm_combined = confusion_matrix(y_test_combined, combined_predictions_or)
    results['cm_combined'] = cm_combined
    results['precision_combined'] = precision_score(y_test_combined, combined_predictions_or)
    results['recall_combined'] = recall_score(y_test_combined, combined_predictions_or)
    results['f1_combined'] = f1_score(y_test_combined, combined_predictions_or)
    results['classification_report_combined'] = classification_report(y_test_combined, combined_predictions_or, target_names=['Normal (0)', 'Tampering (1)'], output_dict=True)


    # --- Compare Individual Model Performance (on the same test set) ---
    print("\n--- Individual Model Performance on Unified Test Set ---")

    # XGBoost Evaluation (customer-level)
    cm_xgb_cust = confusion_matrix(y_test_combined, y_pred_xgb_binary_customer_level)
    results['cm_xgb_cust'] = cm_xgb_cust
    results['precision_xgb_cust'] = precision_score(y_test_combined, y_pred_xgb_binary_customer_level)
    results['recall_xgb_cust'] = recall_score(y_test_combined, y_pred_xgb_binary_customer_level)
    results['f1_xgb_cust'] = f1_score(y_test_combined, y_pred_xgb_binary_customer_level)

    # Autoencoder Evaluation (customer-level)
    cm_ae_cust = confusion_matrix(y_test_combined, y_pred_ae_binary_customer_level)
    results['cm_ae_cust'] = cm_ae_cust
    results['precision_ae_cust'] = precision_score(y_test_combined, y_pred_ae_binary_customer_level)
    results['recall_ae_cust'] = recall_score(y_test_combined, y_pred_ae_binary_customer_level)
    results['f1_ae_cust'] = f1_score(y_test_combined, y_pred_ae_binary_customer_level)

    return results

if __name__ == '__main__':
    # This block will only run if the script is executed directly, not when imported by Streamlit.
    # It allows you to still test the core framework without the GUI.
    print("Running the detection framework directly for testing...")
    results = run_detection_framework()
    if 'error' in results:
        print(results['error'])
    else:
        print("\n--- Detection Framework Results Summary ---")
        print(f"Combined F1-Score: {results['f1_combined']:.4f}")
        print(f"XGBoost F1-Score: {results['f1_xgb_cust']:.4f}")
        print(f"Autoencoder F1-Score: {results['f1_ae_cust']:.4f}")
        # You can print more results here if needed for direct execution