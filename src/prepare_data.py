import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "irrigation_prediction.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COL = 'Irrigation_Need'

def clean_data(df):
    """Fill missing values and remove duplicate rows."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    numeric_cols = cleaned.select_dtypes(include=['number']).columns
    cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median())
    return cleaned

def create_stratified_splits(dataframe):
    """Create balanced train, validation and test datasets."""
    train_df, temp_df = train_test_split(dataframe, test_size=0.30, stratify=dataframe[TARGET_COL], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[TARGET_COL], random_state=42)
    return train_df, val_df, test_df

def run_pipeline():
    """Run the complete Step 2 data pipeline."""
    print("Running SC07 Data Pipeline...")
    raw_data = pd.read_csv(RAW_FILE)
    
    cleaned_data = clean_data(raw_data)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_data, validation_data, test_data = create_stratified_splits(cleaned_data)
    
    # Save files
    cleaned_data.to_csv(PROCESSED_DIR / "irrigation_clean.csv", index=False)
    train_data.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation_data.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_data.to_csv(PROCESSED_DIR / "test.csv", index=False)
    
    print("SC07 STEP 2 DATA PIPELINE COMPLETED")
    print(f"Cleaned dataset: {len(cleaned_data)} rows")
    print(f"Training dataset: {len(train_data)} rows")
    print(f"Output location: {PROCESSED_DIR}")

if __name__ == "__main__":
    run_pipeline()