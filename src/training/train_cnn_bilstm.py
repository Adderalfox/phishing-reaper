import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import json
from src.data_pipeline.preprocess import preprocess_dataset, BYTE_VOCAB_SIZE
from src.models.cnn_bilstm import PhishModel

# ========== Config ==========
CSV_PATH = 'data/Phishing_URL_Dataset.csv'
ARTIFACTS_PATH = 'artifacts/'
MODEL_SAVE_DIR = 'models/'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== Dataset Class ==========
class PhishingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# ========== Train Function ==========  
def train_one_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for inputs, labels in tqdm(dataloader, desc='Training'):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ========== Evaluation Function ==========
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating'):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ========== Main Function ==========
def main():
    print('Preprocessing...')
    X_train, X_val, y_train, y_val = preprocess_dataset(CSV_PATH, mode="train", save_path=ARTIFACTS_PATH)

    train_dataset = PhishingDataset(X_train, y_train)
    val_dataset = PhishingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PhishModel(vocab_size=BYTE_VOCAB_SIZE).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Resume training block
    # checkpoint_path = os.path.join(MODEL_SAVE_DIR, 'model_data_3.pt')
    # if os.path.exists(checkpoint_path):
    #     print(f"Loading model weights from {checkpoint_path}")
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    #     print("Model loaded successfully. Resuming training...")
    # else:
    #     print("No checkpoint found. Training from scratch.")


    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f'\nTraining Loss: {train_loss:.4f}')

        val_accuracy = evaluate(model, val_loader)
        print(f'\nValidation Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, 'model_data_o3.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print('The model has been saved!')
    
    print('Training complete.')

if __name__ == '__main__':
    main()