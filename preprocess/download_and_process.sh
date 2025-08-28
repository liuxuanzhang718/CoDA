#!/bin/bash

# CoDA Data Download and Process Script

echo "Starting data download and processing..."

# Set path
SAVE_PATH="./data"
mkdir -p $SAVE_PATH

# Download files
echo "Downloading files..."
python preprocess/download.py --save_path $SAVE_PATH

# Merge index files
echo "Merging index files..."
cd $SAVE_PATH
cat part_aa part_ab > e5_Flat.index

# Decompress corpus
echo "Decompressing corpus..."
gzip -d wiki-18.jsonl.gz

# Show results
echo "Done! Files saved in: $SAVE_PATH"
ls -lh
