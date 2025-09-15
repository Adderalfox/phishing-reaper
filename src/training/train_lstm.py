import torch
import torch.nn as nn
from src.models.lstm_generator import URLDataset
from src.models.lstm_generator import LSTMGenerator
from torch.utils.data import DataLoader

def train_model(urls, num_epochs=10, seq_length=20, batch_size=16, lr=0.003, model_path="../saved_models/lstm_model.pth"):
    chars = sorted(set("".join(urls)))
    char_to_idx = {c: i+1 for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    dataset = URLDataset(urls, char_to_idx, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Using: {device}")

    model = LSTMGenerator(len(char_to_idx)+1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(x)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char
    }, model_path)
    print(f"Model saved to {model_path}")

    return model, char_to_idx, idx_to_char
