from flask import Flask, request, jsonify
import os
from Bio import SeqIO
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Config
MODEL_FILE = "processed_data/euk_taxonomy_cnn.keras"
MASTER_CSV = "processed_data/eukaryotic_master.csv"
MAX_LEN = 500

# Load model
model = load_model(MODEL_FILE)
classes = pd.read_csv(MASTER_CSV)['taxonomy'].unique()
le = LabelEncoder()
le.fit(classes)

def one_hot_encode(seq, max_len):
    mapping = {'A':0,'C':1,'G':2,'T':3,'N':4}
    arr = np.zeros((max_len,5), dtype=np.float32)
    for i, nuc in enumerate(seq[:max_len]):
        idx = mapping.get(nuc, 4)
        arr[i, idx] = 1
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    sequences = []
    ids = []

    # Read FASTA
    for rec in SeqIO.parse(file, "fasta"):
        sequences.append(str(rec.seq).upper())
        ids.append(rec.id)

    X_test = np.array([one_hot_encode(s, MAX_LEN) for s in sequences])
    pred_probs = model.predict(X_test)
    top_indices = np.argsort(-pred_probs, axis=1)[:, :5]

    results = []
    for i, seq_id in enumerate(ids):
        top_preds = []
        for idx in top_indices[i]:
            label = le.inverse_transform([idx])[0].replace("PREDICTED: ", "")
            conf = round(pred_probs[i][idx]*100, 2)
            top_preds.append({"label": label, "confidence": conf})
        results.append({
            "sequence_id": seq_id,
            "sequence_length": len(sequences[i]),
            "top_predictions": top_preds
        })
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
