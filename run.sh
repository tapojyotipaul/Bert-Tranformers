#!/bin/bash

# Activate environment and install packages
conda init bash

conda install -c conda-forge -y torch
## conda install -c conda-forge -y tensorflow-gpu
pip install tensorflow_hub
pip install bert-for-tf2
git clone https://github.com/tapojyotipaul/Bert-Inferencing
cd Bert-Inferencing
python3 Bert_Inferencing.py