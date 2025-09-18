#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from Bio import SeqIO

# -----------------------------
# Parameters
# -----------------------------
MODEL_FILE = "processed_data/euk_taxonomy_cnn.keras"  # trained model path
MASTER_CSV = "processed_data/euk_taxonomy_master.csv" # master CSV
INPUT_FILE = "test_known.fasta"                        # your test FASTA
MAX_LEN = 500

# -----------------------------
# Load model
# -----------------------------
model = load_model(MODEL_FILE)
print("✅ Model loaded.")

# -----------------------------
# Load label encoder
# -----------------------------
classes = pd.read_csv(MASTER_CSV)['taxonomy'].unique()
le = LabelEncoder()
le.fit(classes)

# -----------------------------
# Read sequences
# -----------------------------
sequences = [str(rec.seq).upper() for rec in SeqIO.parse(INPUT_FILE, "fasta")]
ids = [rec.id for rec in SeqIO.parse(INPUT_FILE, "fasta")]

print(f"Total sequences to test: {len(sequences)}")

# -----------------------------
# One-hot encode
# -----------------------------
def one_hot_encode(seq, max_len):
    mapping = {'A':0,'C':1,'G':2,'T':3,'N':4}
    arr = np.zeros((max_len,5), dtype=np.float32)
    for i, nuc in enumerate(seq[:max_len]):
        idx = mapping.get(nuc, 4)
        arr[i, idx] = 1
    return arr

X_test = np.array([one_hot_encode(s, MAX_LEN) for s in sequences])

# -----------------------------
# Predict taxonomy
# -----------------------------
pred_probs = model.predict(X_test)
pred_labels = le.inverse_transform(np.argmax(pred_probs, axis=1))
confidences = np.max(pred_probs, axis=1)

# -----------------------------
# Show results
# -----------------------------
for seq_id, label, conf in zip(ids, pred_labels, confidences):
    print(f"Sequence ID: {seq_id} | Predicted Taxonomy: {label} | Confidence: {conf:.2f}")

# -----------------------------
# Abundance table
# -----------------------------
df = pd.DataFrame({"taxonomy": pred_labels})
ab_table = df['taxonomy'].value_counts().reset_index()
ab_table.columns = ['taxonomy','count']
ab_table['relative_abundance'] = ab_table['count']/ab_table['count'].sum()

print("\nAbundance Table:\n", ab_table)
