# ========== Base Image ==========
FROM python:3.10-slim

# ========== Set Working Directory ==========
WORKDIR /app

# ========== Copy Project Files ==========
COPY . /app

# ========== Install System Dependencies ==========
RUN apt-get update && apt-get install -y build-essential

# ========== Install Python Dependencies ==========
RUN pip install --no-cache-dir -r requirements.txt

# Runs the full pipeline (LSTM generation + CNN-BiLSTM classification)
CMD ["python", "-m", "scripts.automation"]

