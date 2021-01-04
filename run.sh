#!/bin/bash

# Activate environment and install packages
conda init bash

pip install torch
pip install transformers
git clone https://github.com/tapojyotipaul/Bert-Tranformers
cd Bert-Tranformers
cd 'Pytorch Model Bert'
wget https://tapo1992.s3.us-east-2.amazonaws.com/saved_model/pytorch_model.bin
cd ..
python3 py_run.py
