import torch
from tqdm import tqdm

def train_epoch(model, dataloader, opt, criterion, device, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    for brand_idx, x_in, y_target in tqdm(dataloader, desc='train', leave=False):
        brand_idx = brand_idx.to(device)
        x_in = x_in.to(device)
        y_target = y_target.to(device)
        opt.zero_grad()
        logits, _ = model(brand_idx, x_in)
        B, T, V = logits.size()
        loss = criterion(logits.view(B*T, V), y_target.view(B*T))
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total_loss += loss.item() * B
    return total_loss / len(dataloader.dataset)

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for brand_idx, x_in, y_target in dataloader:
            brand_idx = brand_idx.to(device)
            x_in = x_in.to(device)
            y_target = y_target.to(device)
            logits, _ = model(brand_idx, x_in)
            B, T, V = logits.size()
            loss = criterion(logits.view(B*T, V), y_target.view(B*T))
            total_loss += loss.item() * B
    return total_loss / len(dataloader.dataset)