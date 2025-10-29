"""
Download and parse ClinVar data for missense variants
Save as: download_clinvar.py
"""
import pandas as pd
import requests
from io import StringIO
import re

print("Downloading ClinVar variant summary...")
url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

# Download the file
response = requests.get(url, stream=True)
print(f"Download status: {response.status_code}")

# Read into pandas
df = pd.read_csv(url, sep='\t', compression='gzip', low_memory=False)
print(f"Total variants downloaded: {len(df)}")

# Filter for missense variants with clear pathogenicity
print("\nFiltering for missense variants...")
missense = df[
    (df['Type'] == 'single nucleotide variant') &
    (df['ClinicalSignificance'].isin([
        'Pathogenic', 'Likely pathogenic', 
        'Benign', 'Likely benign'
    ]))
].copy()

print(f"Missense variants with clear labels: {len(missense)}")

# Extract protein change (HGVS p.)
def extract_protein_change(text):
    if pd.isna(text):
        return None
    # Look for p.notation like p.Ala123Val or p.A123V
    match = re.search(r'p\.([A-Z][a-z]{2}\d+[A-Z][a-z]{2}|[A-Z]\d+[A-Z])', str(text))
    return match.group(1) if match else None

missense['ProteinChange'] = missense['Name'].apply(extract_protein_change)
missense = missense[missense['ProteinChange'].notna()]

print(f"Variants with protein notation: {len(missense)}")

# Create binary label (1=Pathogenic, 0=Benign)
missense['Label'] = missense['ClinicalSignificance'].apply(
    lambda x: 1 if 'Pathogenic' in x else 0
)

# Keep relevant columns
final_df = missense[['GeneSymbol', 'ProteinChange', 'ClinicalSignificance', 'Label']].copy()

# Balance dataset (sample equal numbers of each class)
pathogenic = final_df[final_df['Label'] == 1]
benign = final_df[final_df['Label'] == 0]

print(f"\nPathogenic: {len(pathogenic)}, Benign: {len(benign)}")

# Take minimum of both to balance
n_samples = min(len(pathogenic), len(benign))
balanced_df = pd.concat([
    pathogenic.sample(n=min(n_samples, 3000), random_state=42),
    benign.sample(n=min(n_samples, 3000), random_state=42)
])

print(f"\nBalanced dataset size: {len(balanced_df)}")
print(balanced_df['Label'].value_counts())

# Save to CSV
balanced_df.to_csv('clinvar_missense.csv', index=False)
print("\n✅ Saved to clinvar_missense.csv")
print("\nSample data:")
print(balanced_df.head(10))