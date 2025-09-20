import pandas as pd
import re
import json
import os
from sklearn.model_selection import train_test_split
from confusable_homoglyphs import confusables

MAX_LEN = 200
PAD_IDX = 0
UNK_IDX = 1
BYTE_VOCAB_SIZE = 258

def build_vocab():
    char2idx = {i: i + 2 for i in range(256)}
    char2idx["<PAD>"] = 0
    char2idx["<UNK>"] = 1
    
    idx2char = {idx: byte for byte, idx in char2idx.items() if isinstance(byte, int)}
    idx2char[0] = "<PAD>"
    idx2char[1] = "<UNK>"
    return char2idx, idx2char

def encode_url(url, maxlen=MAX_LEN):
    try:
        byte_seq = list(url.encode("utf-8", errors="ignore"))
    except Exception:
        byte_seq = []
    if len(byte_seq) > maxlen:
        byte_seq = byte_seq[:maxlen]

    encoded = [b + 2 if 0 <= b <= 255 else UNK_IDX for b in byte_seq]

    if len(encoded) < maxlen:
        encoded += [PAD_IDX] * (maxlen - len(encoded))

    return encoded

def preprocess_dataset(csv_path, preprocess_file_name, mode, save_path='../artifacts/', maxlen=200, test_size=0.2):
    df = pd.read_csv(csv_path)

    os.makedirs(save_path, exist_ok=True)
    meta_path = os.path.join(save_path, preprocess_file_name)

    if mode == "train":
        char2idx, idx2char = build_vocab()
        meta = {"char2idx": char2idx, "idx2char": idx2char, "maxlen": maxlen}
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        print(f"New vocab built with size {len(char2idx)} and saved to {meta_path}")
    else:
        with open(meta_path) as f:
            meta = json.load(f)
        char2idx = meta["char2idx"]
        idx2char = meta["idx2char"]
        print(f"Loaded existing vocab with size {len(char2idx)} from {meta_path}")

    df['encoded_url'] = df['domain_name'].apply(lambda x: encode_url(x, maxlen))

    if mode == "train":
        X_train, X_val, y_train, y_val = train_test_split(
            df['encoded_url'].tolist(), df['label'].tolist(),
            test_size=test_size, random_state=42
        )
        return X_train, X_val, y_train, y_val
    else:
        return df['encoded_url'].tolist(), df['label'].tolist()