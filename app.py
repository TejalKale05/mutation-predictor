"""
Streamlit web app for mutation impact prediction
Save as: app.py
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils import parse_protein_hgvs, fetch_uniprot_sequence, extract_features
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Mutation Impact Predictor",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('mutation_predictor.joblib')
    except:
        return None

model = load_model()

# Header
st.markdown('<h1 class="main-header">🧬 Protein Mutation Impact Predictor</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
Predict whether a protein mutation is <b>pathogenic</b> (disease-causing) or <b>benign</b> using machine learning
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    ### How to Use:
    1. Enter a **UniProt ID** (e.g., P04637)
    2. Enter a **mutation** in HGVS format
    3. Click **Predict**
    
    ### Mutation Format:
    - `p.A23V` (standard HGVS)
    - `A23V` (short form)
    - `Ala23Val` (three-letter)
    
    ### Example Inputs:
    """)
    
    st.info("**TP53 (P04637)**\n`p.R175H` - Known pathogenic")
    st.info("**TP53 (P04637)**\n`p.P72R` - Known benign")
    st.info("**HBB (P68871)**\n`p.E6V` - Sickle cell")
    
    st.markdown("---")
    st.caption("💡 Find UniProt IDs at [uniprot.org](https://www.uniprot.org)")
    
    if model is None:
        st.error("⚠️ Model not loaded! Run `train_model.py` first.")

# Check if model is loaded
if model is None:
    st.error("❌ Model not found! Please run the following commands:")
    st.code("""
python create_dataset.py
python train_model.py
    """)
    st.stop()

# Main input form
col1, col2 = st.columns([1, 1])

with col1:
    uniprot_id = st.text_input(
        "🔍 UniProt ID",
        value="P04637",
        placeholder="e.g., P04637",
        help="Enter the UniProt accession number"
    )

with col2:
    mutation = st.text_input(
        "🧬 Protein Mutation",
        value="p.R175H",
        placeholder="e.g., p.A23V",
        help="HGVS protein notation"
    )

# Predict button
if st.button("🔮 Predict Mutation Impact", type="primary", use_container_width=True):
    
    if not uniprot_id or not mutation:
        st.warning("⚠️ Please enter both UniProt ID and mutation")
        st.stop()
    
    with st.spinner("🔄 Analyzing mutation..."):
        try:
            # Step 1: Parse mutation
            st.info(f"📝 Parsing mutation: {mutation}")
            ref, pos, alt = parse_protein_hgvs(mutation)
            st.success(f"✅ Parsed as: {ref}{pos}{alt}")
            
            # Step 2: Fetch sequence
            with st.spinner(f"🌐 Fetching protein sequence from UniProt ({uniprot_id})..."):
                seq = fetch_uniprot_sequence(uniprot_id)
            st.success(f"✅ Retrieved sequence: {len(seq)} amino acids")
            
            # Step 3: Verify reference
            if pos > len(seq):
                st.error(f"❌ Position {pos} exceeds protein length ({len(seq)})")
                st.stop()
            
            if seq[pos-1] != ref:
                st.error(f"❌ Reference mismatch at position {pos}")
                st.error(f"Expected: **{ref}**, Found in sequence: **{seq[pos-1]}**")
                st.info("💡 Tip: Check if the mutation notation matches the UniProt sequence")
                st.stop()
            
            st.success(f"✅ Reference verified: {ref} at position {pos}")
            
            # Step 4: Extract features
            with st.spinner("🔬 Extracting molecular features..."):
                features = extract_features(ref, alt, seq, pos)
            
            # Step 5: Make prediction
            X = features.reshape(1, -1)
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            # Results in columns
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric(
                    label="Mutation",
                    value=f"{ref}{pos}{alt}",
                    help="Original → Mutant amino acid"
                )
            
            with res_col2:
                if prediction == 1:
                    st.markdown('<div class="danger-box"><h3>⚠️ PATHOGENIC</h3><p>Likely disease-causing</p></div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box"><h3>✅ BENIGN</h3><p>Likely harmless</p></div>', 
                              unsafe_allow_html=True)
            
            with res_col3:
                confidence = max(probability) * 100
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1f}%",
                    help="Model certainty"
                )
            
            # Probability distribution
            st.markdown("### 📊 Probability Distribution")
            
            prob_df = pd.DataFrame({
                'Class': ['Benign', 'Pathogenic'],
                'Probability': [probability[0], probability[1]]
            })
            
            fig, ax = plt.subplots(figsize=(10, 3))
            colors = ['#28a745' if probability[0] > 0.5 else '#dc3545',
                     '#dc3545' if probability[1] > 0.5 else '#28a745']
            
            bars = ax.barh(prob_df['Class'], prob_df['Probability'], 
                          color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Probability', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_title('Classification Probability', fontsize=14, fontweight='bold')
            
            # Add percentage labels
            for bar, prob in zip(bars, prob_df['Probability']):
                ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.1%}', va='center', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature details
            with st.expander("🔬 Feature Details"):
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("**Substitution Scores:**")
                    st.write(f"- BLOSUM62: `{features[0]:.1f}`")
                    st.write(f"- Grantham Distance: `{features[1]:.1f}`")
                    st.write(f"- Total features: `{len(features)}`")
                
                with detail_col2:
                    st.markdown("**Sequence Context:**")
                    window_size = 11
                    half = window_size // 2
                    start = max(0, pos - 1 - half)
                    end = min(len(seq), pos + half)
                    context = seq[start:end]
                    
                    # Highlight mutated position
                    highlight_pos = min(pos - 1 - start, len(context) - 1)
                    if highlight_pos >= 0 and highlight_pos < len(context):
                        context_display = (
                            context[:highlight_pos] + 
                            f"**[{context[highlight_pos]}→{alt}]**" + 
                            context[highlight_pos+1:]
                        )
                        st.markdown(context_display)
            
            # Disclaimer
            st.markdown("---")
            st.warning("""
            ⚠️ **Important Disclaimer**: 
            - This is a **research tool** for educational purposes only
            - **NOT for clinical diagnosis** or medical decisions
            - Results should be validated by domain experts
            - Always consult healthcare professionals for medical advice
            """)
            
        except ValueError as e:
            st.error(f"❌ Input Error: {str(e)}")
            st.info("💡 Check your UniProt ID and mutation format")
            
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")
            with st.expander("🐛 Debug Information"):
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem;'>
    <p><b>Mutation Impact Predictor</b></p>
    <p>Built with Python • Streamlit • Scikit-learn • BioPython</p>
    <p>Data source: Curated variants • Model: Random Forest Classifier</p>
</div>
""", unsafe_allow_html=True)