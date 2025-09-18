#!/usr/bin/env python3
from Bio import SeqIO
import csv
import os

# Input folder with raw FASTA files
INPUT_DIR = "edna_data"
OUTPUT_DIR = "processed_data"

# Minimum length filter (to remove junk fragments)
MIN_LENGTH = 200

# Make output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_and_extract(fasta_file):
    base = os.path.basename(fasta_file).replace(".fasta", "")
    clean_fasta = os.path.join(OUTPUT_DIR, f"{base}_clean.fasta")
    csv_file = os.path.join(OUTPUT_DIR, f"{base}_metadata.csv")

    with open(clean_fasta, "w") as f_out, open(csv_file, "w", newline="") as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(["id", "sequence", "taxonomy"])  # CSV header

        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq).upper()
            if len(seq) < MIN_LENGTH:
                continue

            # Try to extract taxonomy from description
            desc = record.description
            taxonomy = "Unknown"
            if "[" in desc and "]" in desc:
                taxonomy = desc.split("[")[-1].replace("]", "").strip()

            # Write cleaned FASTA
            f_out.write(f">{record.id}\n{seq}\n")

            # Write CSV
            writer.writerow([record.id, seq, taxonomy])

    print(f"✅ Processed {fasta_file} → {clean_fasta}, {csv_file}")

def main():
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".fasta"):
            clean_and_extract(os.path.join(INPUT_DIR, file))

if __name__ == "__main__":
    main()
