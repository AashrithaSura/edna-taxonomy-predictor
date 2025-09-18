#!/usr/bin/env python3
import os
import glob
import csv
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# -----------------------------
# Parameters
# -----------------------------
INPUT_DIR = os.path.expanduser("~/edna_data")
OUTPUT_DIR = os.path.expanduser("~/processed_data")
MAX_LEN = 500
MIN_LEN = 200
MODEL_FILE = os.path.join(OUTPUT_DIR, "euk_taxonomy_cnn.keras")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Step 1: Preprocess & Merge FASTA
# -----------------------------
def clean_and_extract(fasta_file):
    import os, csv
    from Bio import SeqIO

    base = os.path.basename(fasta_file).replace(".fasta","")
    clean_fasta = os.path.join(OUTPUT_DIR, f"{base}_clean.fasta")
    csv_file = os.path.join(OUTPUT_DIR, f"{base}_metadata.csv")
    
    with open(clean_fasta,"w") as f_out, open(csv_file,"w", newline="") as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(["id","sequence","taxonomy"])
        
        for record in SeqIO.parse(fasta_file,"fasta"):
            seq = str(record.seq).upper()
            if len(seq) < MIN_LEN:
                continue
            
            # Extract taxonomy from header
            desc_parts = record.description.split()
            # take second + third word as genus/species
            if len(desc_parts) >= 3:
                taxonomy = desc_parts[1] + " " + desc_parts[2]
            else:
                taxonomy = "Unknown"
            
            f_out.write(f">{record.id}\n{seq}\n")
            writer.writerow([record.id, seq, taxonomy])
    
    return csv_file


# Process all FASTA
fasta_files = glob.glob(os.path.join(INPUT_DIR, "*.fasta"))
csv_files = [clean_and_extract(f) for f in fasta_files]

# Merge all metadata
df_master = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
# Remove sequences with unknown taxonomy
df_master = df_master[df_master['taxonomy']!="Unknown"]
print(f"Total sequences for training: {len(df_master)}")
print(f"Number of classes: {df_master['taxonomy'].nunique()}")
df_master.to_csv(os.path.join(OUTPUT_DIR,"eukaryotic_master.csv"), index=False)

# -----------------------------
# Step 2: Encode sequences
# -----------------------------
def one_hot_encode(seq,max_len):
    mapping = {'A':0,'C':1,'G':2,'T':3,'N':4}
    arr = np.zeros((max_len,5),dtype=np.float32)
    for i,nuc in enumerate(seq[:max_len]):
        idx = mapping.get(nuc,4)
        arr[i,idx] = 1
    return arr

X = np.array([one_hot_encode(s,MAX_LEN) for s in df_master['sequence']])
labels = df_master['taxonomy'].astype(str)
le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)
y = tf.keras.utils.to_categorical(y,num_classes)

# -----------------------------
# Step 3: Train/test split
# -----------------------------
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# -----------------------------
# Step 4: Build CNN
# -----------------------------
model = Sequential([
    Conv1D(64,7,activation='relu',input_shape=(MAX_LEN,5)),
    MaxPooling1D(3),
    Conv1D(128,5,activation='relu'),
    MaxPooling1D(3),
    Conv1D(256,3,activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(256,activation='relu'),
    Dropout(0.3),
    Dense(num_classes,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# -----------------------------
# Step 5: Train
# -----------------------------
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30,batch_size=128)

# Save model
model.save(MODEL_FILE)
print(f"✅ Model saved: {MODEL_FILE}")

# -----------------------------
# Step 6: Prediction function
# -----------------------------
def predict_sequences(sequences):
    X_new = np.array([one_hot_encode(s.upper(),MAX_LEN) for s in sequences])
    preds = model.predict(X_new)
    pred_labels = le.inverse_transform(np.argmax(preds,axis=1))
    return pred_labels, preds

# Abundance table
def abundance_table(sequences):
    pred_labels,_ = predict_sequences(sequences)
    df = pd.DataFrame(pred_labels,columns=['taxonomy'])
    table = df['taxonomy'].value_counts().reset_index()
    table.columns = ['taxonomy','count']
    table['relative_abundance'] = table['count']/table['count'].sum()
    return table

# Example usage
if __name__=="__main__":
    test_sequences = [
        "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTGACT",
        "TTAGGCAGTCAGTACGATCGATCGATCGTACGAT"
    ]
    labels, _ = predict_sequences(test_sequences)
    print("Predicted Taxonomy:", labels)
    ab_table = abundance_table(test_sequences)
    print("Abundance Table:\n", ab_table)
