"""
Train Random Forest model
Save as: train_model.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    classification_report, confusion_matrix, accuracy_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("TRAINING MUTATION IMPACT PREDICTOR")
print("="*70)


print("\nLoading dataset...")
try:
    X = np.load('X_features.npy')
    y = np.load('y_labels.npy')
except FileNotFoundError:
    print("❌ ERROR: Dataset files not found!")
    print("Please run: python create_dataset.py")
    exit(1)


X = X.astype(np.float32)
y = y.astype(np.int32)

print(f"  Features shape: {X.shape}")
print(f"  Labels shape:   {y.shape}")


if len(X) == 0 or len(y) == 0:
    print("\n❌ ERROR: Dataset is empty!")
    print("Please run: python create_dataset.py")
    exit(1)

if len(X) < 10:
    print(f"\n⚠️  WARNING: Very small dataset ({len(X)} samples)")
    print("Results may not be reliable. Recommend at least 20 samples.")

print(f"\nLabel distribution:")
print(f"  Benign (0):     {np.sum(y==0)}")
print(f"  Pathogenic (1): {np.sum(y==1)}")


print("\nSplitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"  Training set:   {len(X_train)} samples")
print(f"  Test set:       {len(X_test)} samples")


print("\nTraining Random Forest Classifier...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42,
    verbose=0
)

rf.fit(X_train, y_train)
print("✅ Training complete!")


print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)


if len(X_train) >= 5:
    print("\nCross-validation (5-fold)...")
    cv_scores = cross_val_score(rf, X_train, y_train, cv=min(5, len(X_train)), 
                                scoring='roc_auc', n_jobs=-1)
    print(f"  ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")


y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print(f"\nTest Set Performance:")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  ROC-AUC:   {roc_auc:.3f}")
print(f"  PR-AUC:    {pr_auc:.3f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Benign', 'Pathogenic'],
                          digits=3))


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Ben  Path")
print(f"Actual  Benign   {cm[0,0]:3d}   {cm[0,1]:3d}")
print(f"        Path     {cm[1,0]:3d}   {cm[1,1]:3d}")


print("\nTop 10 Most Important Features:")
feature_names = ['BLOSUM62', 'Grantham'] + [f'Window_{i}' for i in range(X.shape[1]-2)]
importances = rf.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]

for idx in top_indices:
    print(f"  {feature_names[idx]:15s}: {importances[idx]:.4f}")


model_file = 'mutation_predictor.joblib'
joblib.dump(rf, model_file)
print(f"\n✅ Model saved: {model_file}")


try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Pathogenic'],
                yticklabels=['Benign', 'Pathogenic'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("📊 Confusion matrix plot saved: confusion_matrix.png")
except Exception as e:
    print(f"⚠️  Could not save plot: {e}")

print("\n" + "="*70)
print("NEXT STEP: Run 'streamlit run app.py' to launch web interface")

print("="*70)
