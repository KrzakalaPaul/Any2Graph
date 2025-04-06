#!/bin/bash

# URL
URL="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

# Download raw files
echo "Downloading raw files"
wget $URL -O data/QM9.csv

# Preprocessing
echo "Preprocessing raw files..."
python ./preprocessing_QM9.py
rm data/QM9.csv