"""
Utility functions for mutation impact prediction.
Robust version (Biopython ≥1.85 compatible)
Save as: utils.py
"""

import re
import requests
import numpy as np


try:
    # For older Biopython versions
    from Bio.SubsMat.MatrixInfo import blosum62
except ImportError:
    # For Biopython ≥1.80
    from Bio.Align import substitution_matrices
    blosum62 = substitution_matrices.load("BLOSUM62")


UNIPROT_API = "https://rest.uniprot.org/uniprotkb/"
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}



def parse_protein_hgvs(hgvs):
    """
    Parse HGVS protein mutation notation.
    Supports: p.A23V, A23V, Ala23Val, p.Ala23Val
    Returns: (ref_aa, position, alt_aa)
    """
    if not isinstance(hgvs, str):
        raise TypeError("HGVS input must be a string")

    hgvs = hgvs.strip()

    
    m = re.match(r'p\.?([A-Z][a-z]{2}|[A-Z])(\d+)([A-Z][a-z]{2}|[A-Z])$', hgvs)
    if not m:
        m2 = re.match(r'^([A-Z])(\d+)([A-Z])$', hgvs)
        if not m2:
            raise ValueError(f"Invalid HGVS format: '{hgvs}'")
        return m2.group(1), int(m2.group(2)), m2.group(3)

    ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)

    
    three_to_one = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
    }

    if len(ref) > 1:
        ref = three_to_one.get(ref, ref)
    if len(alt) > 1:
        alt = three_to_one.get(alt, alt)

    if not (ref.isalpha() and alt.isalpha()):
        raise ValueError(f"Failed to parse mutation: '{hgvs}'")

    return ref, pos, alt



def fetch_uniprot_sequence(uniprot_id):
    """
    Fetch protein sequence from UniProt in FASTA format.
    Returns amino acid sequence as string.
    Raises ValueError if sequence not found.
    """
    if not isinstance(uniprot_id, str) or not uniprot_id.strip():
        raise ValueError("UniProt ID must be a non-empty string")

    url = f"{UNIPROT_API}{uniprot_id}.fasta"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch UniProt sequence: {e}")

    lines = response.text.strip().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Invalid FASTA format for UniProt ID '{uniprot_id}'")

    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    if not seq or any(ch.islower() for ch in seq):
        raise ValueError(f"Invalid sequence returned for '{uniprot_id}'")

    return seq



def blosum_score(a, b):
    """
    Get BLOSUM62 substitution score between two amino acids.
    Returns integer score, or 0 if not found.
    """
    if not a or not b:
        return 0
    a, b = a.upper(), b.upper()
    return blosum62.get((a, b), blosum62.get((b, a), 0))



GRANTHAM = {
    ('S','R'):110, ('S','L'):145, ('S','P'):74, ('S','T'):58, ('S','A'):99,
    ('S','V'):124, ('S','G'):56, ('S','I'):142, ('S','F'):155, ('S','Y'):144,
    ('S','C'):112, ('S','H'):89, ('S','Q'):68, ('S','N'):46, ('S','K'):121,
    ('S','D'):65, ('S','E'):80, ('S','M'):135, ('S','W'):177,
    ('R','L'):102, ('R','P'):103, ('R','T'):71, ('R','A'):112, ('R','V'):96,
    ('R','G'):125, ('R','I'):97, ('R','F'):97, ('R','Y'):77, ('R','C'):180,
    ('R','H'):29, ('R','Q'):43, ('R','N'):86, ('R','K'):26, ('R','D'):96,
    ('R','E'):54, ('R','M'):91, ('R','W'):101,
    ('L','P'):98, ('L','T'):92, ('L','A'):96, ('L','V'):32, ('L','G'):138,
    ('L','I'):5, ('L','F'):22, ('L','Y'):36, ('L','C'):198, ('L','H'):99,
    ('L','Q'):113, ('L','N'):153, ('L','K'):107, ('L','D'):172, ('L','E'):138,
    ('L','M'):15, ('L','W'):61,
    ('P','T'):38, ('P','A'):27, ('P','V'):68, ('P','G'):42, ('P','I'):95,
    ('P','F'):114, ('P','Y'):110, ('P','C'):169, ('P','H'):77, ('P','Q'):76,
    ('P','N'):91, ('P','K'):103, ('P','D'):108, ('P','E'):93, ('P','M'):87,
    ('P','W'):147,
    ('T','A'):58, ('T','V'):69, ('T','G'):59, ('T','I'):89, ('T','F'):103,
    ('T','Y'):92, ('T','C'):149, ('T','H'):47, ('T','Q'):42, ('T','N'):65,
    ('T','K'):78, ('T','D'):85, ('T','E'):65, ('T','M'):81, ('T','W'):128,
    ('A','V'):64, ('A','G'):60, ('A','I'):94, ('A','F'):113, ('A','Y'):112,
    ('A','C'):195, ('A','H'):86, ('A','Q'):91, ('A','N'):111, ('A','K'):106,
    ('A','D'):126, ('A','E'):107, ('A','M'):84, ('A','W'):148,
    ('V','G'):109, ('V','I'):29, ('V','F'):50, ('V','Y'):55, ('V','C'):192,
    ('V','H'):84, ('V','Q'):96, ('V','N'):133, ('V','K'):97, ('V','D'):152,
    ('V','E'):121, ('V','M'):21, ('V','W'):88,
    ('G','I'):135, ('G','F'):153, ('G','Y'):147, ('G','C'):159, ('G','H'):98,
    ('G','Q'):87, ('G','N'):80, ('G','K'):127, ('G','D'):94, ('G','E'):98,
    ('G','M'):127, ('G','W'):184,
    ('I','F'):21, ('I','Y'):33, ('I','C'):198, ('I','H'):94, ('I','Q'):109,
    ('I','N'):149, ('I','K'):102, ('I','D'):168, ('I','E'):134, ('I','M'):10,
    ('I','W'):61,
    ('F','Y'):22, ('F','C'):205, ('F','H'):100, ('F','Q'):116, ('F','N'):158,
    ('F','K'):102, ('F','D'):177, ('F','E'):140, ('F','M'):28, ('F','W'):40,
    ('Y','C'):194, ('Y','H'):83, ('Y','Q'):99, ('Y','N'):143, ('Y','K'):85,
    ('Y','D'):160, ('Y','E'):122, ('Y','M'):36, ('Y','W'):37,
    ('C','H'):174, ('C','Q'):154, ('C','N'):139, ('C','K'):202, ('C','D'):154,
    ('C','E'):170, ('C','M'):196, ('C','W'):215,
    ('H','Q'):24, ('H','N'):68, ('H','K'):32, ('H','D'):81, ('H','E'):40,
    ('H','M'):87, ('H','W'):115,
    ('Q','N'):46, ('Q','K'):53, ('Q','D'):61, ('Q','E'):29, ('Q','M'):101,
    ('Q','W'):130,
    ('N','K'):94, ('N','D'):23, ('N','E'):42, ('N','M'):142, ('N','W'):174,
    ('K','D'):101, ('K','E'):56, ('K','M'):95, ('K','W'):110,
    ('D','E'):45, ('D','M'):160, ('D','W'):181,
    ('E','M'):126, ('E','W'):152, ('M','W'):67,
}


def grantham_distance(a, b):
    """
    Get Grantham distance (physicochemical change between amino acids).
    Returns integer distance (0–215), defaults to 100 if undefined.
    """
    if not a or not b:
        return 100
    if a == b:
        return 0
    return GRANTHAM.get((a, b), GRANTHAM.get((b, a), 100))



def window_onehot(seq, pos, window=11):
    """
    One-hot encode a sequence window centered at the mutation position.
    Returns flat numpy array of shape (window * 20,).
    """
    if not seq or pos < 1 or pos > len(seq):
        raise ValueError("Position out of sequence range")

    half = window // 2
    start = pos - 1 - half
    arr = np.zeros(window * 20, dtype=np.float32)

    for i in range(window):
        idx = start + i
        if 0 <= idx < len(seq):
            aa = seq[idx].upper()
            if aa in AA_INDEX:
                arr[i * 20 + AA_INDEX[aa]] = 1.0

    return arr



def extract_features(ref, alt, seq, pos, window=11):
    """
    Extract combined features for mutation (BLOSUM, Grantham, and one-hot).
    Returns numpy array of features.
    """
    try:
        blosum_val = float(blosum_score(ref, alt))
        grantham_val = float(grantham_distance(ref, alt))
        onehot_vec = window_onehot(seq, pos, window)
        return np.concatenate(([blosum_val, grantham_val], onehot_vec))
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")



if __name__ == "__main__":
    print("✅ utils.py loaded successfully.")
    try:
        ref, pos, alt = parse_protein_hgvs("p.A23V")
        seq = "M" * 50
        print("Ref:", ref, "Pos:", pos, "Alt:", alt)
        print("BLOSUM62:", blosum_score(ref, alt))
        print("Grantham:", grantham_distance(ref, alt))
        print("Feature vector length:", len(extract_features(ref, alt, seq, pos)))
    except Exception as err:
        print("❌ Error:", err)

