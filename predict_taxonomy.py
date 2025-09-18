#!/usr/bin/env python3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------
# Step 0: Parameters
# -----------------------------
MODEL_FILE = "processed_data/euk_taxonomy_cnn.h5"
MASTER_CSV = "processed_data/eukaryotic_master.csv"  # for LabelEncoder classes
MAX_LEN = 500

# -----------------------------
# Step 1: Load trained model
# -----------------------------
model = load_model(MODEL_FILE)
print("✅ Model loaded.")

# -----------------------------
# Step 2: Load LabelEncoder classes
# -----------------------------
df_master = pd.read_csv(MASTER_CSV)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_master['taxonomy'].astype(str))

# -----------------------------
# Step 3: One-hot encode sequences
# -----------------------------
def one_hot_encode(seq, max_len):
    mapping = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4}
    arr = np.zeros((max_len, 5), dtype=np.float32)
    for i, nucleotide in enumerate(seq[:max_len]):
        idx = mapping.get(nucleotide, 4)
        arr[i, idx] = 1
    return arr

def encode_sequences(sequences):
    return np.array([one_hot_encode(s.upper(), MAX_LEN) for s in sequences])

# -----------------------------
# Step 4: Predict taxonomy
# -----------------------------
def predict_taxonomy(sequences):
    X = encode_sequences(sequences)
    preds = model.predict(X)
    pred_labels = le.inverse_transform(np.argmax(preds, axis=1))
    return pred_labels, preds

# -----------------------------
# Step 5: Example usage
# -----------------------------
# Replace this with your eDNA sequences from FASTA/reads
test_sequences = [
    "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTGACT",
    "TTAGGCAGTCAGTACGATCGATCGATCGTACGAT"
]

pred_labels, pred_probs = predict_taxonomy(test_sequences)

for seq, label, prob in zip(test_sequences, pred_labels, pred_probs):
    print(f"Sequence: {seq[:30]}... Predicted Taxonomy: {label} (Confidence: {max(prob):.2f})")

# -----------------------------
# Step 6: Abundance table
# -----------------------------
# If you have many sequences:
def abundance_table(sequences):
    pred_labels, _ = predict_taxonomy(sequences)
    df = pd.DataFrame(pred_labels, columns=['taxonomy'])
    table = df['taxonomy'].value_counts().reset_index()
    table.columns = ['taxonomy', 'count']
    table['relative_abundance'] = table['count'] / table['count'].sum()
    return table

# Example for multiple sequences
ab_table = abundance_table(test_sequences)
print("\nAbundance Table:\n", ab_table)
