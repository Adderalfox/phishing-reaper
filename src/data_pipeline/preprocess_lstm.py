from collections import Counter
import pandas as pd
from torch.utils.data import Dataset
import torch

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'

def build_char_vocab(domains, min_freq=1):
    all_chars = Counter()
    for d in domains:
        all_chars.update(list(d))
    chars = [c for c, f in all_chars.items() if f >= min_freq]
    chars = sorted(chars)
    idx2char = [PAD, SOS, EOS, UNK] + chars
    char2idx = {c:i for i, c in enumerate(idx2char)}
    return char2idx, idx2char

def encode_domain(s, char2idx):
    res = [char2idx.get(SOS)]
    for ch in s:
        res.append(char2idx.get(ch, char2idx.get(UNK)))
    res.append(char2idx.get(EOS))
    return res

def pad_seq(seq, maxlen, pad_idx):
    if len(seq) >= maxlen:
        return seq[:maxlen]
    else:
        return seq + [pad_idx] * (maxlen - len(seq))

class DomainDataset(Dataset):
    def __init__(self, df, char2idx, brand2idx, max_len=80, src_col='Corresponding CSE Domain Name', tgt_col='Identified Phishing/Suspected Domain Name'):
        self.samples = []
        self.char2idx = char2idx
        self.brand2idx = brand2idx
        self.maxlen = max_len
        pad_idx = char2idx[PAD]
        for _, row in df.iterrows():
            brand = str(row[src_col]).strip().lower()
            tgt = str(row[tgt_col]).strip().lower()
            if not brand or not tgt or pd.isna(brand) or pd.isna(tgt):
                continue
            enc = encode_domain(tgt, char2idx)
            enc = pad_seq(enc, max_len, pad_idx)
            self.samples.append((brand2idx[brand], enc))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        brand_idx, seq = self.samples[idx]
        return torch.tensor(brand_idx, dtype=torch.long), torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

        