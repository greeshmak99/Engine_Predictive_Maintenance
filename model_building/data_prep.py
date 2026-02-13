"""
Data Preparation Script - Production Version with New Data Handling
Loads pre-split data from Hugging Face or re-splits if needed
Supports merging new data into existing dataset
"""

import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login, hf_hub_download

# Configuration
DATASET_REPO_ID = "Quantum9999/engine-predictive-maintenance"
TARGET_COLUMN = "Engine Condition"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Control flags
USE_PRESPLIT_DATA = os.environ.get("USE_PRESPLIT_DATA", "true").lower() == "true"
UPLOAD_TO_HF = os.environ.get("UPLOAD_DATA_TO_HF", "false").lower() == "true"
MERGE_NEW_DATA = os.environ.get("MERGE_NEW_DATA", "false").lower() == "true"
NEW_DATA_FILENAME = os.environ.get("NEW_DATA_FILENAME", "pending_data.csv")


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


def check_for_new_data(hf_token):
    """Check if new data file exists on HF"""
    print("=" * 70)
    print("CHECKING FOR NEW DATA")
    print("=" * 70)
    
    try:
        api = HfApi()
        files = api.list_repo_files(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            token=hf_token
        )
        
        has_new_data = NEW_DATA_FILENAME in files
        print(f"Looking for: {NEW_DATA_FILENAME}")
        print(f"✓ New data file {'FOUND' if has_new_data else 'NOT FOUND'}\n")
        return has_new_data
        
    except Exception as e:
        print(f"⚠ Error checking for new data: {e}")
        print(f"Assuming no new data available\n")
        return False


def load_new_data(hf_token):
    """Load new data from HF if available"""
    print("=" * 70)
    print(f"LOADING NEW DATA: {NEW_DATA_FILENAME}")
    print("=" * 70)
    
    try:
        # Download new data file
        new_data_path = hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=NEW_DATA_FILENAME,
            repo_type="dataset",
            token=hf_token
        )
        
        # Load into dataframe
        new_df = pd.read_csv(new_data_path)
        print(f"✓ New data loaded")
        print(f"  Shape: {new_df.shape}")
        print(f"  Columns: {new_df.columns.tolist()}\n")
        
        return new_df
        
    except Exception as e:
        print(f"✗ Error loading new data: {e}\n")
        return None


def merge_datasets(existing_df, new_df):
    """Merge existing and new data, removing duplicates"""
    print("=" * 70)
    print("MERGING DATASETS")
    print("=" * 70)
    
    print(f"Before merge:")
    print(f"  Existing data: {existing_df.shape}")
    print(f"  New data: {new_df.shape}")
    
    # Concatenate
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    print(f"  After concatenation: {merged_df.shape}")
    
    # Remove duplicates
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    duplicates_removed = initial_count - len(merged_df)
    
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Final merged data: {merged_df.shape}")
    
    print(f"\nTarget distribution in merged data:")
    print(f"{merged_df[TARGET_COLUMN].value_counts()}\n")
    
    return merged_df


def load_presplit_data():
    """Load already-split train and test data from Hugging Face"""
    print("=" * 70)
    print("LOADING PRE-SPLIT DATA FROM HUGGING FACE")
    print("=" * 70)
    
    # Load train data
    train_dataset = load_dataset(
        DATASET_REPO_ID,
        data_files="train.csv",
        split="train"
    )
    train_df = train_dataset.to_pandas()
    
    # Load test data
    test_dataset = load_dataset(
        DATASET_REPO_ID,
        data_files="test.csv",
        split="train"
    )
    test_df = test_dataset.to_pandas()
    
    print(f"✓ Pre-split data loaded successfully")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    print(f"\n  Train target distribution:")
    print(f"{train_df[TARGET_COLUMN].value_counts()}")
    print(f"\n  Test target distribution:")
    print(f"{test_df[TARGET_COLUMN].value_counts()}\n")
    
    return train_df, test_df


def load_full_dataset():
    """Load full dataset from HF (for merging or re-splitting)"""
    print("=" * 70)
    print("LOADING FULL DATASET FROM HUGGING FACE")
    print("=" * 70)
    
    try:
        # Try to load from default split
        dataset = load_dataset(DATASET_REPO_ID, split="train")
        df = dataset.to_pandas()
        print(f"✓ Full dataset loaded")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}\n")
        return df
        
    except Exception as e:
        print(f"⚠ Could not load default split, trying train.csv...")
        # Fallback: load train.csv (which might be the full dataset)
        dataset = load_dataset(
            DATASET_REPO_ID,
            data_files="train.csv",
            split="train"
        )
        df = dataset.to_pandas()
        print(f"✓ Dataset loaded from train.csv")
        print(f"  Shape: {df.shape}\n")
        return df


def split_data(df):
    """Perform stratified split on data"""
    print("=" * 70)
    print("PERFORMING STRATIFIED SPLIT")
    print("=" * 70)
    
    # Stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE
    )
    
    print(f"✓ Split completed")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    print(f"\n  Train target distribution:")
    print(f"{train_df[TARGET_COLUMN].value_counts()}")
    print(f"\n  Test target distribution:")
    print(f"{test_df[TARGET_COLUMN].value_counts()}\n")
    
    return train_df, test_df


def save_datasets_locally(train_df, test_df):
    """Save train and test datasets locally"""
    print("=" * 70)
    print("SAVING DATASETS LOCALLY")
    print("=" * 70)
    
    os.makedirs("data", exist_ok=True)
    
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Datasets saved locally")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}\n")
    
    return train_path, test_path


def upload_to_huggingface(train_path, test_path, hf_token, upload_flag):
    """Upload datasets to Hugging Face"""
    if not upload_flag:
        print("=" * 70)
        print("SKIPPING UPLOAD TO HUGGING FACE")
        print("=" * 70)
        print("ℹ  Using existing split on Hugging Face")
        print("ℹ  Set UPLOAD_DATA_TO_HF=true to upload new split\n")
        return
    
    print("=" * 70)
    print("UPLOADING TO HUGGING FACE")
    print("=" * 70)
    
    api = HfApi()
    
    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=hf_token
    )
    print("✓ Train dataset uploaded")
    
    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="test.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=hf_token
    )
    print("✓ Test dataset uploaded\n")


def cleanup_new_data_file(hf_token):
    """Remove pending_data.csv after successful merge"""
    print("=" * 70)
    print("CLEANING UP NEW DATA FILE")
    print("=" * 70)
    
    try:
        api = HfApi()
        api.delete_file(
            path_in_repo=NEW_DATA_FILENAME,
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            token=hf_token
        )
        print(f"✓ {NEW_DATA_FILENAME} removed from HF\n")
    except Exception as e:
        print(f"⚠ Could not remove {NEW_DATA_FILENAME}: {e}\n")


def main():
    """Main execution pipeline"""
    # CRITICAL FIX: Declare global at the very start of function
    global UPLOAD_TO_HF
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION PIPELINE")
    print("=" * 70)
    print(f"Use pre-split data: {USE_PRESPLIT_DATA}")
    print(f"Merge new data: {MERGE_NEW_DATA}")
    print(f"Upload to HF: {UPLOAD_TO_HF}")
    print("=" * 70 + "\n")
    
    # Authenticate
    hf_token = authenticate_hf()
    
    # Track if we should upload (use local variable to avoid global issues)
    should_upload = UPLOAD_TO_HF
    
    # Check workflow mode
    if USE_PRESPLIT_DATA and not MERGE_NEW_DATA:
        # STANDARD MODE: Load existing train/test from HF
        print("📌 MODE: Using pre-split data (efficient)\n")
        train_df, test_df = load_presplit_data()
        
    elif MERGE_NEW_DATA:
        # NEW DATA MODE: Merge new data with existing, then re-split
        print("📌 MODE: Merging new data and re-splitting\n")
        
        # Check if new data exists
        has_new_data = check_for_new_data(hf_token)
        
        if not has_new_data:
            print("⚠ WARNING: MERGE_NEW_DATA=true but no new data found!")
            print(f"ℹ  Expected file: {NEW_DATA_FILENAME}")
            print(f"ℹ  Falling back to re-splitting existing data\n")
            
            full_df = load_full_dataset()
            train_df, test_df = split_data(full_df)
        else:
            # Load existing full dataset
            existing_df = load_full_dataset()
            
            # Load new data
            new_df = load_new_data(hf_token)
            
            if new_df is not None:
                # Merge datasets
                merged_df = merge_datasets(existing_df, new_df)
                
                # Split merged data
                train_df, test_df = split_data(merged_df)
                
                # Auto-enable upload since we have new data
                should_upload = True
                print("ℹ  Auto-enabled upload due to new data merge\n")
            else:
                print("⚠ Could not load new data, using existing dataset\n")
                train_df, test_df = split_data(existing_df)
    else:
        # RE-SPLIT MODE: Load full dataset and re-split
        print("📌 MODE: Re-splitting existing data\n")
        full_df = load_full_dataset()
        train_df, test_df = split_data(full_df)
    
    # Save locally for pipeline
    train_path, test_path = save_datasets_locally(train_df, test_df)
    
    # Upload to HF (if needed)
    upload_to_huggingface(train_path, test_path, hf_token, should_upload)
    
    # Cleanup new data file if merge was successful
    if MERGE_NEW_DATA and should_upload:
        cleanup_new_data_file(hf_token)
    
    print("=" * 70)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  Final train size: {len(train_df)}")
    print(f"  Final test size: {len(test_df)}")
    print(f"  Data uploaded to HF: {should_upload}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
