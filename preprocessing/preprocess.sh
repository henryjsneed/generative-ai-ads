#!/bin/bash
rm -rf ../../data/test/*
rm -rf ../../data/validation/*
rm -rf ../../data/train/*
echo "Cleaned old data"
python3 generate_simulated_data.py
python3 split_dataset.py
echo "Finished generating simulated data"