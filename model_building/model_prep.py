"""
Model Training and Registration Script - Production Version
Trains XGBoost model and optionally updates on Hugging Face Model Hub
"""

import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from huggingface_hub import HfApi, login, hf_hub_download

# Configuration
DATA_PATH = "data"
TARGET_COLUMN = "Engine Condition"
MODEL_FILENAME = "xgb_tuned_model.joblib"
HF_MODEL_REPO = "Quantum9999/xgb-predictive-maintenance"

# FIXED: Match actual dataset column names
FEATURE_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]

# Control flags
TRAIN_NEW_MODEL = os.environ.get("TRAIN_NEW_MODEL", "true").lower() == "true"
UPLOAD_MODEL_TO_HF = os.environ.get("UPLOAD_MODEL_TO_HF", "false").lower() == "true"
COMPARE_WITH_EXISTING = os.environ.get("COMPARE_WITH_EXISTING", "true").lower() == "true"
DATA_WAS_UPDATED = os.environ.get("DATA_WAS_UPDATED", "false").lower() == "true"


def authenticate_hf():
    """Authenticate with Hugging Face"""
    print("=" * 70)
    print("AUTHENTICATING WITH HUGGING FACE")
    print("=" * 70)
    
    hf_token = os.environ.get("HF_EN_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_EN_TOKEN environment variable not found")
    
    login(token=hf_token)
    print("✓ Successfully authenticated\n")
    return hf_token


def adjust_workflow_for_data_update():
    """Automatically adjust flags if data was updated"""
    global UPLOAD_MODEL_TO_HF, COMPARE_WITH_EXISTING
    
    if DATA_WAS_UPDATED:
        print("=" * 70)
        print("DATA UPDATE DETECTED - ADJUSTING WORKFLOW")
        print("=" * 70)
        print("ℹ  New data detected, automatically adjusting pipeline:")
        print("  - TRAIN_NEW_MODEL: true (training on new data)")
        print("  - UPLOAD_MODEL_TO_HF: true (new model needed)")
        print("  - COMPARE_WITH_EXISTING: false (different datasets)")
        print("=" * 70 + "\n")
        
        UPLOAD_MODEL_TO_HF = True
        COMPARE_WITH_EXISTING = False


def load_prepared_data():
    """Load train and test datasets"""
    print("=" * 70)
    print("STEP 1: LOADING PREPARED DATA")
    print("=" * 70)
    
    train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
    test_df = pd.read_csv(f"{DATA_PATH}/test.csv")
    
    print(f"✓ Data loaded successfully")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    print(f"\n  Actual columns: {train_df.columns.tolist()}\n")
    
    return train_df, test_df


def prepare_features(train_df, test_df):
    """Prepare feature matrices and target vectors"""
    print("=" * 70)
    print("STEP 2: PREPARING FEATURES")
    print("=" * 70)
    
    missing_cols = [col for col in FEATURE_COLUMNS if col not in train_df.columns]
    if missing_cols:
        print(f"✗ ERROR: Missing columns: {missing_cols}")
        print(f"Available: {train_df.columns.tolist()}")
        raise KeyError(f"Missing columns: {missing_cols}")
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    
    print(f"✓ Features prepared")
    print(f"  Number of features: {len(FEATURE_COLUMNS)}")
    print(f"  Features: {FEATURE_COLUMNS}\n")
    
    return X_train, X_test, y_train, y_test


def load_existing_model(hf_token):
    """Load existing model from Hugging Face for comparison"""
    print("=" * 70)
    print("LOADING EXISTING MODEL FROM HUGGING FACE")
    print("=" * 70)
    
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=MODEL_FILENAME,
            token=hf_token
        )
        existing_model = joblib.load(model_path)
        print(f"✓ Existing model loaded from HF")
        print(f"  Repository: {HF_MODEL_REPO}")
        print(f"  File: {MODEL_FILENAME}\n")
        return existing_model
    except Exception as e:
        print(f"ℹ  No existing model found: {e}")
        print(f"ℹ  Will train and upload new model\n")
        return None


def train_xgboost_model(X_train, y_train):
    """Train XGBoost model with tuned hyperparameters"""
    print("=" * 70)
    print("TRAINING XGBOOST MODEL")
    print("=" * 70)
    
    print("Hyperparameters:")
    hyperparams = {
        'n_estimators': 250,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'gamma': 0.5,
        'reg_alpha': 0.5,
        'reg_lambda': 7,
        'min_child_weight': 10
    }
    
    for param, value in hyperparams.items():
        print(f"  - {param}: {value}")
    
    model = XGBClassifier(
        subsample=0.8,
        reg_lambda=7,
        reg_alpha=0.5,
        n_estimators=250,
        min_child_weight=10,
        max_depth=6,
        learning_rate=0.01,
        gamma=0.5,
        colsample_bytree=0.6,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("\n🔄 Training in progress...")
    model.fit(X_train, y_train)
    print("✓ Model training completed\n")
    
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    print("-" * 70)
    print(f"EVALUATING: {model_name}")
    print("-" * 70)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("Performance Metrics:")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}\n")
    
    return metrics


def compare_models(existing_metrics, new_metrics):
    """Compare existing and new model performance"""
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<15} {'Existing':<12} {'New':<12} {'Improvement':<12}")
    print("-" * 70)
    
    improved = False
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        existing_val = existing_metrics[metric]
        new_val = new_metrics[metric]
        diff = new_val - existing_val
        symbol = "✓" if diff > 0 else "✗" if diff < 0 else "="
        
        print(f"{metric:<15} {existing_val:<12.4f} {new_val:<12.4f} {diff:+.4f} {symbol}")
        
        if diff > 0.001:
            improved = True
    
    print("\n" + "=" * 70)
    if improved:
        print("✓ NEW MODEL SHOWS IMPROVEMENT")
    else:
        print("ℹ  NEW MODEL SIMILAR TO EXISTING")
    print("=" * 70 + "\n")
    
    return improved


def save_model_locally(model):
    """Save model to local file"""
    print("=" * 70)
    print("SAVING MODEL LOCALLY")
    print("=" * 70)
    
    os.makedirs("model", exist_ok=True)
    model_path = f"model/{MODEL_FILENAME}"
    
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}\n")
    
    return model_path


def upload_model_to_hf(model_path, hf_token):
    """Upload model to Hugging Face Model Hub"""
    if not UPLOAD_MODEL_TO_HF:
        print("=" * 70)
        print("SKIPPING UPLOAD TO HUGGING FACE")
        print("=" * 70)
        print("ℹ  Using existing model on Hugging Face")
        print("ℹ  Set UPLOAD_MODEL_TO_HF=true to upload\n")
        return
    
    print("=" * 70)
    print("UPLOADING MODEL TO HUGGING FACE")
    print("=" * 70)
    
    api = HfApi()
    
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=MODEL_FILENAME,
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            token=hf_token
        )
        print(f"✓ Model uploaded successfully")
        print(f"  Repository: {HF_MODEL_REPO}\n")
    except Exception as e:
        print(f"⚠ Upload error: {e}\n")


def main():
    """Main execution pipeline"""
    print("\n" + "=" * 70)
    print("MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Train new model: {TRAIN_NEW_MODEL}")
    print(f"Upload to HF: {UPLOAD_MODEL_TO_HF}")
    print(f"Compare with existing: {COMPARE_WITH_EXISTING}")
    print(f"Data was updated: {DATA_WAS_UPDATED}")
    print("=" * 70 + "\n")
    
    adjust_workflow_for_data_update()
    
    hf_token = authenticate_hf()
    train_df, test_df = load_prepared_data()
    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df)
    
    existing_model = None
    existing_metrics = None
    
    if COMPARE_WITH_EXISTING:
        existing_model = load_existing_model(hf_token)
        if existing_model:
            existing_metrics = evaluate_model(
                existing_model, X_test, y_test, 
                model_name="Existing Model (from HF)"
            )
    
    if TRAIN_NEW_MODEL:
        print("=" * 70)
        print("STEP 3: TRAINING NEW MODEL")
        print("=" * 70 + "\n")
        new_model = train_xgboost_model(X_train, y_train)
        new_metrics = evaluate_model(new_model, X_test, y_test, model_name="Newly Trained Model")
        
        if existing_metrics and new_metrics and COMPARE_WITH_EXISTING:
            compare_models(existing_metrics, new_metrics)
        
        model_to_save = new_model
    else:
        model_to_save = existing_model
    
    if model_to_save:
        model_path = save_model_locally(model_to_save)
        upload_model_to_hf(model_path, hf_token)
    
    print("=" * 70)
    print("MODEL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
