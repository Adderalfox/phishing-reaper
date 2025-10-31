import os
import torch
import json
import pandas as pd
import requests
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.models.cnn_bilstm import PhishModel
from src.data_pipeline.preprocess import encode_url, MAX_LEN, BYTE_VOCAB_SIZE
from src.training.train_cnn_bilstm import PhishingDataset

# ========== Argument Parser ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Phishing URL Detection Inference Script")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--model", default="src/models/model_data_o4.pt", help="Path to trained model file")
    parser.add_argument("--output", default="results/url_predictions.csv", help="Path to save output predictions")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    return parser.parse_args()

# ========== Config ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    )
}

# ========== URL Check ==========
def is_url_reachable(url: str) -> bool:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        response = requests.head(url, headers=HEADERS, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.RequestException:
        return False

# ========== Preprocessing ==========
def preprocess_single_url(url):
    encoded = encode_url(url, maxlen=MAX_LEN)
    return torch.tensor(encoded, dtype=torch.long)

# ========== Dataset ==========
class URLDataset(Dataset):
    def __init__(self, urls):
        self.urls = urls

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        x = preprocess_single_url(url)
        return x, url

# ========== Prediction ==========
def predict(model, dataloader):
    model.eval()
    results = []

    with torch.no_grad():
        for inputs, urls in tqdm(dataloader, desc="Predicting"):
            if isinstance(inputs, list):
                inputs = torch.stack(inputs)

            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for url, pred in zip(urls, preds):
                results.append({
                    "Domain": url,
                    "Prediction": "Phishing" if pred == 1 else "Legitimate"
                })

    return results

# ========== Main ==========
def main():
    args = parse_args()
    csv_path = args.csv
    model_path = args.model
    output_path = args.output
    batch_size = args.batch_size

    print("Loading model...")
    model = PhishModel(vocab_size=BYTE_VOCAB_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    urls = df.iloc[:, 0].dropna().tolist()

    print(f"Checking reachability for {len(urls)} URLs...")
    valid_urls = []
    for url in tqdm(urls, desc="Verifying URLs"):
        if is_url_reachable(url):
            valid_urls.append(url)

    print(f"{len(valid_urls)} valid URLs found (status 200).")

    if not valid_urls:
        print("No reachable URLs found. Exiting.")
        return

    dataset = URLDataset(valid_urls)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Running predictions...")
    results = predict(model, dataloader)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

if __name__ == "__main__":
    main()
