import torch.nn as nn
import torch.nn.functional as F
import torch

class URLDataset(Dataset):
    def __init__(self, urls, char_to_idx, seq_length=20):
        self.data = []
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
        for url in urls:
            encoded = [char_to_idx.get(c, 0) for c in url]
            if len(encoded) < seq_length:
                encoded += [0] * (seq_length - len(encoded))
            else:
                encoded = encoded[:seq_length]
            self.data.append(torch.tensor(encoded, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]

class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64, num_layers=2):
        super(LSTMGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
        