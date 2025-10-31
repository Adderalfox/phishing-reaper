# 🛡️ Phishing Detection Tool (Dockerised)

## 🚀 Overview
This tool predicts phishing domains using:
- **LSTM Generator** (candidate URL generation)
- **CNN-BiLSTM Classifier** (phishing detection)
- **Transformer Model** (CSE mapping)

There are **two modes** of operation:

---

### 🔹 1. LSTM Generation + Classification
Runs the full automated pipeline — generates URLs using the LSTM model, classifies them, and outputs predictions.

**Command (inside Docker):**
```bash
python -m scripts.automation

## Output:
```
results/url_predictions.csv
```

## 🔹 2. Direct Classification (Dataset Input)
You can directly classify URLs without generation by running:
```bash
python -m src.inference.inference_cnnbilstm --csv data/raw/Shortlisting_Dataset_part_1.csv
python -m src.inference.inference_cnnbilstm --csv data/raw/Shortlisting_Dataset_part_2.csv
```

## 🔹 3. Transformer-Based Mapping
After classification, map results to corresponding CSEs:
```bash
python -m inference_transformer --csv results/url_predictions.csv
```

## 🧱 Environment Requirements
- Python 3.10+
- Docker 20.10+
- All dependencies listed in `requirements.txt`

## 🐳 Running with Docker

### 1️⃣ Build the Image
```bash
docker build -t phish-reaper .
```

### 2️⃣ Run the Automated Pipeline
```bash
docker run phish-reaper
```

### 3️⃣ Run Direct Inference (Example)
```bash
docker run phish-reaper python -m src.inference.inference_cnnbilstm --csv data/raw/Shortlisting_Dataset_part_1.csv
```

### 4️⃣ Run Transformer Mapping
```bash
docker run phish-reaper python -m inference_transformer --csv results/url_predictions.csv
```

## 🧠 Models Used
- **CNN-BiLSTM** for phishing URL classification
- **LSTM Generator** for candidate URL synthesis
- **Transformer** for CSE mapping

## 🧪 Output
Final phishing domain predictions are saved in:
```
results/url_predictions.csv
```

## 🧾 License
© 2025 Ritick Kalita. For evaluation use only.

---

## 📄 **instructions.txt**
*(Plain text for submission — same info as README but simplified.)*

### Instructions to Execute the Dockerised Phishing Detection Tool

**Paste the Datasets in data\raw directory**
```bash
data\raw\Shortlisting_Dataset_part_1.csv
data\raw\Shortlisting_Dataset_part_2.csv
```

**Build the Docker image:**
```bash
docker build -t phish-reaper .
```

**Run the default pipeline (LSTM generation + classification):**
```bash
docker run phish-reaper
```

**Run classification on dataset part 1:**
```bash
docker run phish-reaper python -m src.inference.inference_cnnbilstm --csv data/raw/Shortlisting_Dataset_part_1.csv
```

**Run classification on dataset part 2:**
```bash
docker run phish-reaper python -m src.inference.inference_cnnbilstm --csv data/raw/Shortlisting_Dataset_part_2.csv
```

**Run transformer mapping:**
```bash
docker run phish-reaper python -m inference_transformer --csv results/url_predictions.csv
```

**Output files will be stored in:**
```
results/url_predictions.csv
```