import torch.nn as nn
import torch.nn.functional as F
import torch

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
        