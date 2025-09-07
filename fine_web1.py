"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

from calendar import prmonth
import os
import multiprocessing as mp
from re import I
from httpx import stream
import numpy as np
from sympy import true
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B2"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname("hello"), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    #print(f"this is eot {eot}")
    #print(f"This is the doc:{doc["prompt"]}") # the special <|endoftext|> token delimits all documents
    #print(f"This is the encoding:{enc.encode(doc["prompt"])}")
    tokens.extend(enc.encode_ordinary(doc["text"]))
    #print(f"This is extended token {tokens}")
    #print(f"this the np array:{np.array(tokens)}")
    tokens_np = np.array(tokens)
    #print(f"This is token np {tokens_np}")
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    #print(f"this is token_int {tokens_np_uint16}")
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    print(f"filename:{filename}, token:{tokens_np}")
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
print(f"The cpu count is {nprocs}")
#print(f"The lengrth of fw is:{fw}")
if __name__ =="__main__":
 with mp.Pool(nprocs) as pool:
    print("I am in")
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    # print(pool)
    for tokens in pool.imap(tokenize,iterable=fw,chunksize=16):
        #print(f"im in too with token len{tokens}")

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            #print("hey")
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            #print("hello")
            split = "val" if shard_index == 0 else "train"
            #print(f"shared index:{shared_index},split:{split}")
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb2_{split}_{shard_index:06d}")
            #print(f"Filename:{filename}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            #print(f"The shared size is {shard_size}")
            #print(f"The token count is {token_count}")
            remainder = shard_size - token_count
            #print(f"The remainder is {remainder}")
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {remainder}")
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            print(f"the remainder and token is {remainder},{len(tokens)}")
            try:
              all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            except:
               print("Tokenizing is done")
               break
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])