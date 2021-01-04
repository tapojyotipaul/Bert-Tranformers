# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:02:14 2020

@author: tapojyoti.paul
"""

import torch
import tqdm

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup
import random

import boto3

#s3_client = boto3.resource('s3')    
#KEY = "tapo1992/saved_model/pytorch_model.bin"
#s3_client.Bucket("tb-lumen").download_file(KEY, '/home/ubuntu/Bert-Tranformers/Pytorch Model Bert/pytorch_model.bin')## Loading Dataset
print("_______________________________________________________________")
print("Loading Data...........")
cols = ["sentiment", "id", "date", "query", "user", "text"]
data = pd.read_csv(
    r"test.csv",
    header=None,
    names=cols,
    engine="python",
    encoding="latin1"
)
data.drop(["id", "date", "query", "user"],
          axis=1,
          inplace=True)


print("_______________________________________________________________")
print("Data Pre-processing...........")
def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Removing the @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Removing the URL links
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    # Keeping only letters
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Removing additional whitespaces
    tweet = re.sub(r" +", ' ', tweet)
    return tweet

data_clean = [clean_tweet(tweet) for tweet in data.text]
labels = data.sentiment.values
labels[labels == 4] = 1

from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split

output_dir = r'Pytorch Model Bert'
# Load a trained model and vocabulary that you have fine-tuned
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)
# Copy the model to the GPU.
model.to(device)

X = data_clean
def get_test_data(size: int = 1):
    """Generates a test dataset of the specified size""" 
    num_rows = len(X)
    test_df = X.copy()

    while num_rows < size:
        test_df = test_df + test_df
        num_rows = len(test_df)

    return test_df[:size]

def calculate_stats(time_list):
    """Calculate mean and standard deviation of a list"""
    time_array = np.array(time_list)

    median = np.median(time_array)
    mean = np.mean(time_array)
    std_dev = np.std(time_array)
    max_time = np.amax(time_array)
    min_time = np.amin(time_array)
    quantile_10 = np.quantile(time_array, 0.1)
    quantile_90 = np.quantile(time_array, 0.9)

    basic_key = ["median","mean","std_dev","min_time","max_time","quantile_10","quantile_90"]
    basic_value = [median,mean,std_dev,min_time,max_time,quantile_10,quantile_90]

    dict_basic = dict(zip(basic_key, basic_value))
    
    return pd.DataFrame(dict_basic, index = [0])

import argparse
import logging

from pathlib import Path
from timeit import default_timer as timer

NUM_LOOPS = 50

def run_inference(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    test_twt = get_test_data(num_observations)
    num_rows = len(test_twt)
    print(f"running data prep and inference for {num_rows} sentence(s)..")
    
    run_times = []
    bert_times = []
    prep_time_wo_berts = []
    prep_time_alls = []
    prep_inf_times = []
    inference_times = []
    
    for _ in range(NUM_LOOPS):
        
#######################################################################################################################
        st_tm_bert = timer()
        input_ids = []
        attention_masks = []
        # For every sentence...
        for sent in tqdm.tqdm(test_twt,desc ="Progress.."):
            encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 64,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        batch_size = 100  
        # Create the DataLoader.
        prediction_data = TensorDataset(input_ids, attention_masks,)
        prediction_dataloader = DataLoader(prediction_data,batch_size=batch_size)
        
        end_tm_bert = timer()
        start_time = timer()
        logit_list = []
        for batch in prediction_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
#             print("/////////////")
#             print(len(b_input_ids))
#             print("/////////////")
#######################################################################################################################
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logit_list.append(logits)
        end_time = timer()
#######################################################################################################################

        total_time = end_time - start_time
        run_times.append(total_time*10e3)
        
        bert_time = (end_tm_bert-st_tm_bert)*(10e6)/num_rows
        prep_time_wo_bert = (start_time-end_tm_bert)*(10e6)/num_rows
        prep_time_all = (start_time-st_tm_bert)*(10e6)/num_rows
        inference_time = total_time*(10e6)/num_rows
        prep_inf_time = (end_time-st_tm_bert)*(10e6)/num_rows
        
        bert_times.append(bert_time)
        prep_time_wo_berts.append(prep_time_wo_bert)
        prep_time_alls.append(prep_time_all)
        prep_inf_times.append(prep_inf_time)
        inference_times.append(inference_time)
        
    print("length of predicted df", len(logit_list))
    
    df1 = calculate_stats(bert_times)
    df1["Flag"] = "Only Bert"
    df2 = calculate_stats(prep_time_wo_berts)
    df2["Flag"] = "Prep w/o Bert"
    df3 = calculate_stats(prep_time_alls)
    df3["Flag"] = "Prep with Bert"
    df4 = calculate_stats(prep_inf_times)
    df4["Flag"] = "Prep & Inf Time Total"
    df5 = calculate_stats(inference_times)
    df5["Flag"] = "Inference Time"

    dfs = pd.concat([df1,df2,df3,df5,df4])
    
    print(num_observations, ", ", dfs)
    return dfs
    
STATS = '#, median, mean, std_dev, min_time, max_time, quantile_10, quantile_90'

print("_______________________________________________________________")
print("Inferencing Started...........")
if __name__=='__main__':
    ob_ct = 1  # Start with a single observation
    logging.info(STATS)
    temp_df = pd.DataFrame()
    while ob_ct <= 10000:
        temp = run_inference(ob_ct)
        temp["No_of_Observation"] = ob_ct
        temp_df = temp_df.append(temp)
        ob_ct *= 10
    print("Summary........")
    print(temp_df)
    
    


