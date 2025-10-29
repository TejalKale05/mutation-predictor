# Mutation-predictor

**Author:** Tejal Kale

---

## Overview

The Protein Mutation Impact Predictor is a machine learning-based web application that predicts whether a protein mutation is pathogenic (disease-causing) or benign (harmless). This tool uses Random Forest classification with molecular features extracted from protein sequences, providing predictions with confidence scores through an interactive Streamlit web interface.

## Project Background

Protein mutations play critical roles in human health and disease. Single amino acid substitutions can dramatically alter protein function, causing conditions from cancer to genetic disorders. However, many mutations are neutral polymorphisms with no functional impact. This project implements a supervised machine learning approach that learns from known pathogenic and benign mutations to predict novel variant impacts, offering a faster alternative to expensive experimental characterization methods.

## Technical Approach

The predictor uses a Random Forest classifier trained on curated variants from well-studied genes including TP53, BRCA1, HBB, and CFTR. Feature engineering extracts BLOSUM62 substitution scores (evolutionary conservation), Grantham distances (physicochemical changes), and one-hot encoded sequence windows (local context). This combination provides the model with evolutionary signals, property changes, and positional patterns to distinguish damaging from neutral mutations.

## Installation and Setup

Ensure Python 3.7 or higher is installed. Create a project directory called mutation-predictor and navigate into it. Create five project files: requirements.txt, utils.py, create_dataset.py, train_model.py, and app.py using the provided code. Install dependencies by running "pip install -r requirements.txt" which installs NumPy, Pandas, Scikit-learn, BioPython, Requests, Joblib, Streamlit, Matplotlib, Seaborn, and tqdm. Installation takes 2-5 minutes.

## Step-by-Step Execution Guide

### Step 1: Build the Training Dataset

Run "python create_dataset.py" to create the training data. This script processes approximately 30 curated mutations with known pathogenicity. For each variant, it parses the mutation notation, queries UniProt's REST API for the protein sequence, verifies the reference amino acid matches the sequence, and extracts molecular features. The script saves X_features.npy (features), y_labels.npy (labels), and dataset_metadata.csv (variant information). This process takes 5-10 minutes and requires internet connectivity. Expect 25-28 successful variants out of 30 attempted.

### Step 2: Train the Machine Learning Model

Run "python train_model.py" to train the classifier. The script loads the dataset, splits it 80-20 for training and testing with stratification, and trains a Random Forest with 200 trees and balanced class weights. Training completes in 1-2 minutes. The script evaluates performance using cross-validation and test set metrics including accuracy, ROC-AUC, and PR-AUC, displays a confusion matrix, shows feature importances, and saves the trained model as mutation_predictor.joblib. Expect test accuracy above 70% and ROC-AUC above 0.75.

### Step 3: Launch the Web Server

Run "streamlit run app.py" to start the web application. Streamlit initializes a local server at http://localhost:8501 and automatically opens your browser. The app loads the trained model and provides an interactive interface where users enter a UniProt ID and mutation notation, then receive predictions with color-coded results (red for pathogenic, green for benign), probability distributions, and detailed feature information. The server runs continuously in your terminal until you stop it with Ctrl+C.

## Using the Web Application

Enter a UniProt accession ID (e.g., P04637 for TP53) and mutation in HGVS format (e.g., p.R175H, A23V, or Ala23Val). Click "Predict Mutation Impact" and the app will parse the mutation, fetch the protein sequence, verify the reference amino acid, extract features, and display predictions with confidence scores and probability charts. The sidebar provides example inputs including TP53 p.R175H (pathogenic), TP53 p.P72R (benign), and HBB p.E6V (sickle cell).

## Project Files

The project consists of five core files: requirements.txt lists all Python dependencies, utils.py contains functions for parsing mutations and extracting features, create_dataset.py builds the training dataset from curated variants, train_model.py trains and evaluates the Random Forest classifier, and app.py provides the Streamlit web interface. Generated files include X_features.npy and y_labels.npy (dataset arrays), dataset_metadata.csv (variant details), mutation_predictor.joblib (trained model), and confusion_matrix.png (performance visualization).

## Troubleshooting

If you encounter "Cannot fetch UniProt entry" errors, check your internet connection or wait 30 seconds as UniProt may be rate-limiting. For module import errors, install missing packages with pip. If the web server won't start due to port conflicts, use "streamlit run app.py --server.port 8502" to specify an alternative port. If dataset creation extracts zero features, verify that utils.py exists in the same directory and that you have internet connectivity for API requests.

## Important Disclaimer

This tool is designed for research and educational purposes only. It is not intended for clinical diagnosis or medical decision-making. Predictions should be validated by domain experts and experimental methods. Always consult qualified healthcare professionals for medical advice regarding genetic variants and disease risk.

---

**Project Timeline:** Complete setup and deployment takes approximately 20-30 minutes including installation (5 min), dataset creation (10 min), model training (2 min), and testing (5 min).
