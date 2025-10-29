"""
Check what files exist and their contents
Save as: check_files.py
"""
import os
import numpy as np
import pandas as pd

print("Checking project files...\n")

files = [
    'clinvar_missense.csv',
    'X_features.npy',
    'y_labels.npy',
    'dataset_metadata.csv'
]

for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"✅ {f} exists ({size:,} bytes)")
        
        # Check contents
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(f)
                print(f"   → {len(df)} rows, columns: {list(df.columns)[:5]}")
            except Exception as e:
                print(f"   → Error reading: {e}")
        
        elif f.endswith('.npy'):
            try:
                data = np.load(f)
                print(f"   → Shape: {data.shape}, dtype: {data.dtype}")
            except Exception as e:
                print(f"   → Error reading: {e}")
    else:
        print(f"❌ {f} NOT FOUND")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if not os.path.exists('clinvar_missense.csv'):
    print("❌ Run: python download_clinvar.py")
elif not os.path.exists('X_features.npy'):
    print("❌ Run: python build_dataset.py")
else:
    X = np.load('X_features.npy')
    y = np.load('y_labels.npy')
    if len(X) == 0:
        print("❌ Dataset is empty! Need to rebuild.")
        print("   Issue: build_dataset.py didn't extract features successfully")
    else:
        print(f"✅ Dataset ready: {len(X)} samples")