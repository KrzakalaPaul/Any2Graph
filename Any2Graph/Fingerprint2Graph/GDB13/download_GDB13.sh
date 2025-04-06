#!/bin/bash

# URL
URL="https://zenodo.org/record/5172018/files/GDB13_Subset-ABCDEFGH.smi.gz?download=1"

# Download raw files
echo "Downloading raw files"
wget $URL -O data/GDB13.smi.gz
gunzip data/GDB13.smi.gz

# Preprocessing
echo "Preprocessing raw files..."
python ./preprocessing_GDB13.py
rm data/GDB13.smi