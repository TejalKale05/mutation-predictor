"""
Create training dataset from manually curated variants
This is faster and more reliable than parsing ClinVar
Save as: create_dataset.py
"""
import numpy as np
import pandas as pd
from utils import parse_protein_hgvs, fetch_uniprot_sequence, extract_features
import time

# Curated list of well-known variants with UniProt IDs
# Format: (Gene, UniProtID, Mutation, Label, Description)
TRAINING_VARIANTS = [
    # TP53 (Tumor suppressor) - P04637
    ('TP53', 'P04637', 'p.R175H', 1, 'Pathogenic - common cancer mutation'),
    ('TP53', 'P04637', 'p.R248W', 1, 'Pathogenic - DNA binding loss'),
    ('TP53', 'P04637', 'p.R273H', 1, 'Pathogenic - hotspot mutation'),
    ('TP53', 'P04637', 'p.R282W', 1, 'Pathogenic - loss of function'),
    ('TP53', 'P04637', 'p.P72R', 0, 'Benign - common polymorphism'),
    ('TP53', 'P04637', 'p.P47S', 0, 'Benign - neutral variant'),
    
    # HBB (Hemoglobin) - P68871
    ('HBB', 'P68871', 'p.E6V', 1, 'Pathogenic - sickle cell disease'),
    ('HBB', 'P68871', 'p.E6K', 1, 'Pathogenic - hemoglobin variant'),
    ('HBB', 'P68871', 'p.V2M', 0, 'Benign - normal variant'),
    
    # CFTR (Cystic fibrosis) - P13569
    ('CFTR', 'P13569', 'p.G542X', 1, 'Pathogenic - CF mutation'),
    ('CFTR', 'P13569', 'p.R117H', 0, 'Benign - mild variant'),
    ('CFTR', 'P13569', 'p.M470V', 0, 'Benign - polymorphism'),
    
    # BRCA1 (Breast cancer) - P38398
    ('BRCA1', 'P38398', 'p.C61G', 1, 'Pathogenic - cancer predisposition'),
    ('BRCA1', 'P38398', 'p.P871L', 0, 'Benign - neutral'),
    ('BRCA1', 'P38398', 'p.K1183R', 0, 'Benign - polymorphism'),
    
    # APOE (Alzheimer risk) - P02649
    ('APOE', 'P02649', 'p.C130R', 1, 'Pathogenic - AD risk'),
    ('APOE', 'P02649', 'p.R158C', 0, 'Benign - common variant'),
    
    # F5 (Factor V) - P12259
    ('F5', 'P12259', 'p.R506Q', 1, 'Pathogenic - thrombophilia'),
    ('F5', 'P12259', 'p.H1299R', 0, 'Benign - neutral'),
    
    # GBA (Gaucher disease) - P04062
    ('GBA', 'P04062', 'p.N370S', 1, 'Pathogenic - Gaucher'),
    ('GBA', 'P04062', 'p.E326K', 0, 'Benign - polymorphism'),
    
    # LDLR (Cholesterol) - P01130
    ('LDLR', 'P01130', 'p.C95Y', 1, 'Pathogenic - familial hypercholesterolemia'),
    ('LDLR', 'P01130', 'p.T705I', 0, 'Benign - neutral'),
    
    # SOD1 (ALS) - P00441
    ('SOD1', 'P00441', 'p.A4V', 1, 'Pathogenic - ALS mutation'),
    ('SOD1', 'P00441', 'p.D90A', 0, 'Benign - recessive/polymorphism'),
    
    # HEXA (Tay-Sachs) - P06865
    ('HEXA', 'P06865', 'p.R178H', 1, 'Pathogenic - Tay-Sachs'),
    ('HEXA', 'P06865', 'p.R247W', 1, 'Pathogenic - enzyme deficiency'),
    
    # Additional variants for balance
    ('EGFR', 'P00533', 'p.L858R', 1, 'Pathogenic - cancer driver'),
    ('KRAS', 'P01116', 'p.G12D', 1, 'Pathogenic - oncogenic'),
    ('PTEN', 'P60484', 'p.R130Q', 1, 'Pathogenic - loss of function'),
    ('RB1', 'P06400', 'p.R661W', 1, 'Pathogenic - tumor suppressor loss'),
]

print("="*70)
print("CREATING TRAINING DATASET")
print("="*70)
print(f"\nTotal variants to process: {len(TRAINING_VARIANTS)}")
print("This will take ~5-10 minutes (fetching sequences from UniProt)\n")

features_list = []
labels = []
metadata = []
failed = []

for i, (gene, uniprot_id, mutation, label, description) in enumerate(TRAINING_VARIANTS, 1):
    print(f"[{i}/{len(TRAINING_VARIANTS)}] {gene:8} {mutation:12} ", end='')
    
    try:
        # Parse mutation
        ref, pos, alt = parse_protein_hgvs(mutation)
        
        # Fetch sequence
        seq = fetch_uniprot_sequence(uniprot_id)
        
        # Verify reference amino acid
        if pos > len(seq):
            print(f"❌ Position {pos} exceeds length {len(seq)}")
            failed.append((gene, mutation, "Position out of range"))
            continue
        
        if seq[pos-1] != ref:
            print(f"❌ Ref mismatch: expected {seq[pos-1]}, got {ref}")
            failed.append((gene, mutation, f"Ref mismatch: {seq[pos-1]} vs {ref}"))
            continue
        
        # Extract features
        feats = extract_features(ref, alt, seq, pos)
        
        features_list.append(feats)
        labels.append(label)
        metadata.append({
            'Gene': gene,
            'UniProtID': uniprot_id,
            'Mutation': mutation,
            'Label': label,
            'Description': description
        })
        
        label_str = "PATH" if label == 1 else "BEN "
        print(f"✅ {label_str}")
        
        # Be nice to UniProt API
        time.sleep(0.2)
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        failed.append((gene, mutation, str(e)))
        continue

print("\n" + "="*70)
print("RESULTS")
print("="*70)

if len(features_list) == 0:
    print("\n❌ ERROR: No features extracted!")
    print("\nPossible issues:")
    print("  1. No internet connection")
    print("  2. UniProt API is down")
    print("  3. utils.py file is missing or has errors")
    print("\nFailed variants:")
    for gene, mut, reason in failed:
        print(f"  - {gene} {mut}: {reason}")
    exit(1)

# Convert to numpy arrays
X = np.array(features_list, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print(f"\n✅ Successfully processed: {len(features_list)}/{len(TRAINING_VARIANTS)} variants")
print(f"\nDataset shape:")
print(f"  Features (X): {X.shape}")
print(f"  Labels (y):   {y.shape}")
print(f"  Features per sample: {X.shape[1]}")

# Label distribution
n_benign = np.sum(y == 0)
n_pathogenic = np.sum(y == 1)
print(f"\nLabel distribution:")
print(f"  Benign (0):     {n_benign} ({n_benign/len(y)*100:.1f}%)")
print(f"  Pathogenic (1): {n_pathogenic} ({n_pathogenic/len(y)*100:.1f}%)")

# Save files
print("\nSaving files...")
np.save('X_features.npy', X)
np.save('y_labels.npy', y)

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv('dataset_metadata.csv', index=False)

print("\n✅ Files saved:")
print("   - X_features.npy")
print("   - y_labels.npy")
print("   - dataset_metadata.csv")

if failed:
    print(f"\n⚠️  Failed to process {len(failed)} variants:")
    for gene, mut, reason in failed[:5]:
        print(f"   - {gene} {mut}: {reason}")
    if len(failed) > 5:
        print(f"   ... and {len(failed)-5} more")

print("\n" + "="*70)
print("NEXT STEP: Run 'python train_model.py'")
print("="*70)