"""
SC07 Smart Irrigation System - Step 3 ANN Model Training Script
This script trains a Feed-Forward Artificial Neural Network to predict irrigation needs.
It validates data, trains the model, generates performance metrics, and securely exports artifacts.
"""

from pathlib import Path
import json
import pickle
import time
import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. PATH CONFIGURATION & HYPERPARAMETERS
# ---------------------------------------------------------
print("[INFO] Initializing environment and configuring paths...")
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for folder in [REPORT_DIR, MODEL_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "Irrigation_Need"
FEATURES = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"
]
RANDOM_SEED = 42

hidden_layers = (16, 8)
learning_rate = 0.001
max_epochs = 400

# ---------------------------------------------------------
# 2. UTILITY FUNCTIONS
# ---------------------------------------------------------
def load_split(name):
    """Load a dataset split, validate features, and ensure no missing values."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Cannot find '{path.name}'.")
    
    frame = pd.read_csv(path)
    required = FEATURES + [TARGET]
    
    missing_cols = [col for col in required if col not in frame.columns]
    if missing_cols:
         raise ValueError(f"[ERROR] Missing expected columns in {name}.csv: {missing_cols}")
         
    if frame[required].isna().sum().sum() > 0:
        raise ValueError(f"[ERROR] '{name}.csv' contains missing values.")
        
    return frame

def metric_row(y_true, y_pred, split_name):
    """Calculates classification metrics for a dataset split."""
    return {
        "split": split_name.upper(),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
    }

def show_confusion_matrix(y_true, y_pred, split_name):
    """Computes and exports the confusion matrix visualization."""
    matrix = confusion_matrix(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(6, 5))
    
    image = axis.imshow(matrix, cmap="Greens")
    figure.colorbar(image, ax=axis)
    
    axis.set(
        title=f"ANN Confusion Matrix: {split_name.title()} Split",
        xlabel="Predicted Label", 
        ylabel="Actual Label"
    )
    
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            val = matrix[row, column]
            color = "white" if val > threshold else "black"
            axis.text(column, row, str(val), ha="center", va="center", color=color, fontweight='bold')
            
    figure.tight_layout()
    file_path = RESULTS_DIR / f"step3_{split_name}_confusion_matrix.png"
    figure.savefig(file_path, dpi=120, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# 3. DATA LOADING & VALIDATION
# ---------------------------------------------------------
print("[INFO] Loading and validating processed datasets...")
train_data = load_split("train")
validation_data = load_split("validation")
test_data = load_split("test")

X_train, y_train = train_data[FEATURES], train_data[TARGET]
X_validation, y_validation = validation_data[FEATURES], validation_data[TARGET]
X_test, y_test = test_data[FEATURES], test_data[TARGET]

assert X_train.shape[1] == len(FEATURES), "[ERROR] Feature dimension mismatch detected!"

# ---------------------------------------------------------
# 4. MODEL INITIALIZATION & TRAINING
# ---------------------------------------------------------
print("[INFO] Initializing Artificial Neural Network Pipeline...")
ann_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ann", MLPClassifier(
            hidden_layer_sizes=hidden_layers, 
            activation="relu", 
            solver="adam",
            learning_rate_init=learning_rate, 
            max_iter=max_epochs, 
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=RANDOM_SEED
        ))
    ]
)

print(f"[INFO] Commencing model training on {X_train.shape[0]:,} samples...")
start_time = time.time()
ann_model.fit(X_train, y_train)
elapsed_time = time.time() - start_time
trained_ann = ann_model.named_steps["ann"]

print(f"[SUCCESS] Convergence achieved in {elapsed_time:.2f} seconds (Iterations: {trained_ann.n_iter_}).")

# ---------------------------------------------------------
# 5. EVALUATION & VISUALIZATION
# ---------------------------------------------------------
print("[INFO] Generating predictions and evaluating performance...")
validation_predictions = ann_model.predict(X_validation)
test_predictions = ann_model.predict(X_test)

val_metrics = metric_row(y_validation, validation_predictions, "validation")
test_metrics = metric_row(y_test, test_predictions, "test")
metrics_df = pd.DataFrame([val_metrics, test_metrics])

print("\n" + "-" * 55)
print("COMPREHENSIVE PERFORMANCE SUMMARY")
print("-" * 55)
print(metrics_df.to_string(index=False))
print("-" * 55 + "\n")

print("[INFO] Generating diagnostic visualizations...")
# Loss Curve
losses = trained_ann.loss_curve_
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(losses) + 1), losses, marker='o', markersize=2, color='#1f77b4', linewidth=1.5)
plt.title("ANN Convergence: Training Loss per Iteration", pad=15, fontweight='bold')
plt.xlabel("Training Iteration (Epoch)")
plt.ylabel("Loss Error Margin")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "step3_ann_loss_curve.png", dpi=120, bbox_inches='tight')
plt.close()

# Confusion Matrices
show_confusion_matrix(y_validation, validation_predictions, "validation")
show_confusion_matrix(y_test, test_predictions, "test")

# ---------------------------------------------------------
# 6. ARTIFACT SERIALIZATION
# ---------------------------------------------------------
print("[INFO] Initiating artifact serialization and export protocol...")
metrics_df.to_csv(REPORT_DIR / "step3_ann_metrics.csv", index=False)

for split_name, frame, preds in [("validation", validation_data, validation_predictions), ("test", test_data, test_predictions)]:
    output = frame.copy()
    output["ann_prediction"] = preds
    output.to_csv(REPORT_DIR / f"step3_{split_name}_predictions.csv", index=False)

with (MODEL_DIR / "ann_smart_irrigation.pkl").open("wb") as model_file:
    pickle.dump(ann_model, model_file)

metadata = {
    "project": "SC07 Smart Irrigation System",
    "technique": "Feed-forward ANN / backpropagation",
    "architecture": f"{len(FEATURES)} inputs -> {hidden_layers} -> 1 output",
    "features": FEATURES, 
    "target": TARGET, 
    "random_seed": RANDOM_SEED,
    "export_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
(MODEL_DIR / "ann_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("[SUCCESS] All model artifacts and Step 3 evidence files securely saved.")