import torch.nn as nn
import torch

class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, char_emb_dim, brand_count, brand_emb_dim, hidden_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
        self.brand_emb = nn.Embedding(brand_count, brand_emb_dim)
        input_dim = char_emb_dim + brand_emb_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.lin = nn.Linear(hidden_dim, vocab_size)

    def forward(self, brand_idx, x_chars, hidden=None):
        b_emb = self.brand_emb(brand_idx)
        b_emb_exp = b_emb.unsqueeze(1).expand(-1, x_chars.size(1), -1)
        c_emb = self.char_emb(x_chars)
        inp = torch.cat([c_emb, b_emb_exp], dim=-1)
        out, hidden = self.lstm(inp, hidden)
        logits = self.lin(out)
        return logits, hidden