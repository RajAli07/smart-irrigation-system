"""
SC07 Smart Irrigation System - Step 2 Data Preparation Script
This script automates the ingestion, cleansing, and stratified partitioning 
of the raw agricultural dataset to prepare it for machine learning pipelines.
"""

import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "irrigation_prediction.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COL = 'Irrigation_Need'

# ---------------------------------------------------------
# PIPELINE FUNCTIONS
# ---------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scrub the dataset by removing exact duplicates and applying median 
    imputation for missing numerical values to ensure robustness against outliers.
    """
    print("[INFO] Initiating data cleansing and imputation protocol...")
    cleaned = df.copy()
    
    # Capture metrics for logs
    duplicates_before = cleaned.duplicated().sum()
    
    # Remove duplicates
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    print(f"[INFO] Purged {duplicates_before} duplicate records.")
    
    # Median Imputation
    numeric_cols = cleaned.select_dtypes(include=['number']).columns
    missing_before = cleaned[numeric_cols].isna().sum().sum()
    
    if missing_before > 0:
        print(f"[INFO] Applying median imputation to resolve {missing_before} missing numerical values...")
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median())
    else:
        print("[INFO] No missing numerical values detected. Imputation skipped.")
        
    # EXTRA EFFORT: Defensive Sanity Checks
    assert cleaned.duplicated().sum() == 0, "[CRITICAL] Duplicate removal failed."
    assert cleaned[numeric_cols].isna().sum().sum() == 0, "[CRITICAL] Imputation failed."
    
    return cleaned

def create_stratified_splits(dataframe: pd.DataFrame):
    """
    Create balanced Train (70%), Validation (15%), and Test (15%) splits 
    using stratified sampling on the target variable.
    """
    print(f"[INFO] Performing stratified splitting on target: '{TARGET_COL}'...")
    
    train_df, temp_df = train_test_split(
        dataframe, test_size=0.30, stratify=dataframe[TARGET_COL], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[TARGET_COL], random_state=42
    )
    return train_df, val_df, test_df

def run_pipeline():
    """Execute the complete Step 2 automated data preparation pipeline."""
    print("=" * 55)
    print("SC07 SYSTEM LOG: AUTOMATED DATA PREPARATION PIPELINE")
    print("=" * 55)
    
    # EXTRA EFFORT: File existence validation
    if not RAW_FILE.exists():
        print(f"[ERROR] Raw data file not found at: {RAW_FILE}")
        sys.exit(1)
        
    try:
        print(f"[INFO] Loading raw dataset from {RAW_FILE.name}...")
        raw_data = pd.read_csv(RAW_FILE)
        print(f"[SUCCESS] Loaded {len(raw_data):,} records successfully.")
        
        cleaned_data = clean_data(raw_data)
        
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_data, val_data, test_data = create_stratified_splits(cleaned_data)
        
        print("[INFO] Serializing partitioned datasets...")
        cleaned_data.to_csv(PROCESSED_DIR / "irrigation_clean.csv", index=False)
        train_data.to_csv(PROCESSED_DIR / "train.csv", index=False)
        val_data.to_csv(PROCESSED_DIR / "validation.csv", index=False)
        test_data.to_csv(PROCESSED_DIR / "test.csv", index=False)
        
        # EXTRA EFFORT: Executive Audit Summary
        print("\n" + "=" * 55)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 55)
        print(f" -> Master Cleaned Data:   {len(cleaned_data):,} records")
        print(f" -> Training Split:        {len(train_data):,} records (70%)")
        print(f" -> Validation Split:      {len(val_data):,} records (15%)")
        print(f" -> Testing Split:         {len(test_data):,} records (15%)")
        print(f" -> Serialized Output Dir: {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
        print("=" * 55)
        print("[SUCCESS] Data pipeline execution completed securely.")
        
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()