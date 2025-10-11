import torch
import torch.nn.functional as F
import os
import random
from src.models.lstm_generator import LSTMGenerator
from src.training.train_lstm import train_epoch, eval_epoch
from src.data_pipeline.preprocess_lstm import build_char_vocab, encode_domain, pad_seq, DomainDataset
import argparse
from collections import Counter
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def top_k_logits(logits, k):
    if k <= 0:
        return logits    
    v, ix = torch.topk(logits, k)
    if logits.dim() == 1:
        minv = v[-1]
    else:
        minv = v[:, -1].unsqueeze(1)
    
    return torch.where(logits < minv, torch.full_like(logits, -1e10), logits)

def top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -1e10
    return logits

def sample_domains(model, device, char2idx, idx2char, brand2idx, brand_name, max_len=60, temperature=1.0, top_k=40, top_p=0.9, n_samples=50, seed_prefix='', dedupe=True, legit_list=None):
    model.eval()
    inv = idx2char
    results = []
    pad_idx = char2idx[PAD]
    sos_idx = char2idx[SOS]
    eos_idx = char2idx[EOS]

    brand_idx = torch.tensor([brand2idx.get(brand_name.lower(), 0)], dtype=torch.long, device=device)
    seen = set()
    attempts = 0
    while len(results) < n_samples and attempts < n_samples * 15:
        attempts += 1
        seq = [sos_idx]
        for ch in seed_prefix:
            seq.append(char2idx.get(ch, char2idx[UNK]))
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        hidden = None
        generated = []
        for t in range(max_len):
            logits, hidden = model(brand_idx, seq_tensor, hidden)
            last_logits = logits[0, -1, :].clone()
            last_logits = last_logits / max(temperature, 1e-8)
            # top-k
            if top_k and top_k > 0:
                last_logits = top_k_logits(last_logits, top_k)
            # top-p
            if top_p and top_p < 1.0:
                last_logits = top_p_filtering(last_logits, top_p=top_p)
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == eos_idx:
                break
            generated.append(next_token)
            seq_tensor = torch.cat([seq_tensor, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
        txt = ''.join(inv[idx] for idx in generated)
        txt = txt.strip('.')
        # basic filtering
        if not txt:
            continue
        if legit_list and txt in legit_list:
            continue
        if dedupe and txt in seen:
            continue
        # simple url-safe check
        if not all((c.isalnum() or c in '-.') for c in txt):
            continue
        if len(txt) < 4 or len(txt) > 60:
            continue
        seen.add(txt)
        results.append(f"http://{txt}")
        results.append(f"https://{txt}")
    return results

# -----------------------
# Main: train + save
# -----------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    df = pd.read_csv(args.csv_path)
    src_col = 'Corresponding CSE Domain Name'
    tgt_col = 'Identified Phishing/Suspected Domain Name'
    df['Corresponding CSE Domain Name'] = df['Corresponding CSE Domain Name'].astype(str).str.lower().str.strip()
    df['Identified Phishing/Suspected Domain Name'] = df['Identified Phishing/Suspected Domain Name'].astype(str).str.lower().str.strip()

    domains = df['Identified Phishing/Suspected Domain Name'].dropna().astype(str).tolist()
    char2idx, idx2char = build_char_vocab(domains, min_freq=1)
    vocab_size = len(idx2char)
    print("Vocab size:", vocab_size)

    # build brand2idx
    brands = sorted(df['Corresponding CSE Domain Name'].unique().tolist())
    brand2idx = {b: i for i, b in enumerate(brands)}
    print("Brands:", len(brands))

    if args.mode == 'train':
        dataset = DomainDataset(df, char2idx, brand2idx, max_len=args.max_len, src_col=src_col, tgt_col=tgt_col)
        n = len(dataset)
        if n == 0:
            raise RuntimeError("Empty dataset — check csv and column names.")
        n_train = int(n * 0.9)
        indices = list(range(n)); random.shuffle(indices)
        train_idx = indices[:n_train]; val_idx = indices[n_train:]
        train_ds = Subset(dataset, train_idx); val_ds = Subset(dataset, val_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        model = LSTMGenerator(vocab_size=vocab_size, char_emb_dim=args.char_emb, brand_count=len(brands), brand_emb_dim=args.brand_emb, hidden_dim=args.hidden, n_layers=args.layers, dropout=args.dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=char2idx[PAD])

        best_val = 1e9
        os.makedirs(args.save_dir, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, opt, criterion, device, grad_clip=args.grad_clip)
            val_loss = eval_epoch(model, val_loader, criterion, device)
            print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'model_state': model.state_dict(),
                    'char2idx': char2idx,
                    'idx2char': idx2char,
                    'brand2idx': brand2idx,
                    'args': vars(args)
                }, os.path.join(args.save_dir, 'best_model.pt'))
                print("Saved best model.")
    elif args.mode == 'generate':
        ckpt = torch.load(os.path.join(args.save_dir, 'best_model.pt'), map_location=device)
        model = LSTMGenerator(vocab_size=len(ckpt['idx2char']), char_emb_dim=args.char_emb, brand_count=len(ckpt['brand2idx']), brand_emb_dim=args.brand_emb, hidden_dim=args.hidden, n_layers=args.layers, dropout=args.dropout).to(device)
        model.load_state_dict(ckpt['model_state'])
        char2idx = ckpt['char2idx']
        idx2char = ckpt['idx2char']
        brand2idx = ckpt['brand2idx']
        # build legit blacklist from dataset (don't output these)
        legit_list = set(df[tgt_col].tolist()) | set(df[src_col].tolist())
        samples = sample_domains(model, device, char2idx, idx2char, brand2idx, args.demo_brand, max_len=args.max_len, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, n_samples=args.n_samples, seed_prefix=args.seed_prefix, dedupe=True, legit_list=legit_list)
        if args.out_csv:
            out_df = pd.DataFrame({'domain': samples, 'brand': [args.demo_brand]*len(samples)})
            out_df.to_csv(args.out_csv, index=False)
            print("Saved generated domains to", args.out_csv)
    else:
        raise ValueError("Unknown mode. use --mode train or --mode generate")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='data/processed/Dataset.csv', help='CSV path with required columns')
    parser.add_argument('--save_dir', type=str, default='models', help='where to save model')
    parser.add_argument('--mode', type=str, default='train', choices=['train','generate'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--char_emb', type=int, default=128)
    parser.add_argument('--brand_emb', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--max_len', type=int, default=80)
    parser.add_argument('--demo_brand', type=str, default='airtel.in', help='brand for demo generation (one of CSV brands)')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed_prefix', type=str, default='', help='optional starting prefix characters for generation')
    parser.add_argument('--augment', action='store_true', help='enable runtime augmentation (cheap)')
    parser.add_argument('--aug_per_sample', type=int, default=0, help='how many augmented variants to add per sample during training')
    parser.add_argument('--out_csv', type=str, default='', help='where to dump generated domains (only in generate mode)')
    args = parser.parse_args()
    main(args)