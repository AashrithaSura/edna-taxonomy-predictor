#!/usr/bin/env python3
from Bio import SeqIO
import csv
import os
import glob

# Input folder with raw FASTA files
INPUT_DIR = os.path.expanduser("~/edna_data")
OUTPUT_DIR = os.path.expanduser("~/processed_data")
MASTER_CSV = os.path.join(OUTPUT_DIR, "eukaryotic_master.csv")

MIN_LENGTH = 200  # minimum sequence length filter

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_csv_files = []

def clean_and_extract(fasta_file):
    base = os.path.basename(fasta_file).replace(".fasta", "")
    clean_fasta = os.path.join(OUTPUT_DIR, f"{base}_clean.fasta")
    csv_file = os.path.join(OUTPUT_DIR, f"{base}_metadata.csv")
    all_csv_files.append(csv_file)

    with open(clean_fasta, "w") as f_out, open(csv_file, "w", newline="") as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(["id", "sequence", "taxonomy"])

        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq).upper()
            if len(seq) < MIN_LENGTH:
                continue

            # Extract taxonomy from header if exists
            desc = record.description
            taxonomy = "Unknown"
            if "[" in desc and "]" in desc:
                taxonomy = desc.split("[")[-1].replace("]", "").strip()

            f_out.write(f">{record.id}\n{seq}\n")
            writer.writerow([record.id, seq, taxonomy])

    print(f"✅ Processed {fasta_file} → {clean_fasta}, {csv_file}")

def merge_csvs(csv_files, master_csv):
    with open(master_csv, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "sequence", "taxonomy"])
        for file in csv_files:
            with open(file, "r") as f:
                next(f)  # skip header
                for line in f:
                    writer.writerow(line.strip().split(","))
    print(f"✅ All metadata merged → {master_csv}")

def main():
    fasta_files = glob.glob(os.path.join(INPUT_DIR, "*.fasta"))
    for fasta in fasta_files:
        clean_and_extract(fasta)

    merge_csvs(all_csv_files, MASTER_CSV)

if __name__ == "__main__":
    main()
