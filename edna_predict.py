# edna_predict.py
#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from Bio import SeqIO, Entrez
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

Entrez.email = "surak638@gmail.com"

# -----------------------------
# Load model and classes
# -----------------------------
MODEL_FILE = "processed_data/euk_taxonomy_cnn.keras"
MASTER_CSV = "processed_data/eukaryotic_master.csv"  # make sure the name matches
MAX_LEN = 500

model = load_model(MODEL_FILE)
classes = pd.read_csv(MASTER_CSV)['taxonomy'].unique()
le = LabelEncoder()
le.fit(classes)

# -----------------------------
# One-hot encode function
# -----------------------------
def one_hot_encode(seq, max_len):
    mapping = {'A':0,'C':1,'G':2,'T':3,'N':4}
    arr = np.zeros((max_len,5), dtype=np.float32)
    for i, nuc in enumerate(seq[:max_len]):
        idx = mapping.get(nuc, 4)
        arr[i, idx] = 1
    return arr

# -----------------------------
# Prediction function
# -----------------------------
def predict_sequences(input_file, top_n=5):
    sequences, ids = [], []

    if input_file.lower().endswith(('.fasta', '.fa')):
        for rec in SeqIO.parse(input_file, "fasta"):
            sequences.append(str(rec.seq).upper())
            ids.append(rec.id)
    elif input_file.lower().endswith('.csv'):
        df = pd.read_csv(input_file)
        for i, row in df.iterrows():
            sequences.append(str(row['sequence']).upper())
            ids.append(row['id'] if 'id' in df.columns else f"Seq{i+1}")
    else:  # plain text
        with open(input_file) as f:
            for i, line in enumerate(f):
                sequences.append(line.strip().upper())
                ids.append(f"Seq{i+1}")

    # One-hot encode
    X_test = np.array([one_hot_encode(s, MAX_LEN) for s in sequences])

    # Predict
    pred_probs = model.predict(X_test)
    top_indices = np.argsort(-pred_probs, axis=1)[:, :top_n]

    results = {}
    for i, seq_id in enumerate(ids):
        seq_results = {}
        for idx in top_indices[i][:top_n]:
            label = le.inverse_transform([idx])[0].replace("PREDICTED: ", "")
            conf = pred_probs[i][idx]*100
            seq_results[label] = conf
        results[seq_id] = seq_results

    return results

# -----------------------------
# Optional: fetch NCBI annotation
# -----------------------------
def fetch_annotation(accession):
    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
        record = handle.read()
        handle.close()
        lines = [line for line in record.split("\n") if line.startswith("DEFINITION") or line.startswith("SOURCE")]
        if lines:
            return " | ".join(lines)[:200]
        else:
            return "No annotation found"
    except Exception as e:
        return f"Annotation fetch failed: {e}"
