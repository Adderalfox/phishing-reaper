import os
import re
import time
import requests
import pandas as pd
import tldextract
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Phishing URL Detection Inference Script")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="results/domain_mappings.csv", help="Path to save output predictions")
    return parser.parse_args()

# ========== CONFIG ==========
MAX_WORKERS = 10
REQUEST_TIMEOUT = 10
RETRY_ATTEMPTS = 2

CONTEXT_METHOD = 'embedding'
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CONFIDENCE_THRESHOLD = 0.4  # below this, label will be "None"
LOW_SCORE_LABEL = "None"

# ========== Reference context list ==========
CSE_LIST = {
    "Airtel": "Airtel is an Indian telecommunications company offering mobile services, broadband, and digital TV.",
    "Bank of Baroda": "Bank of Baroda is a major Indian public sector bank providing banking and financial services.",
    "Civil Registration System, MHA": "Civil Registration System is an Indian government portal for births, deaths, and civil registrations (MHA).",
    "HDFC Bank": "HDFC Bank is a leading private sector bank in India providing retail and corporate banking.",
    "ICICI Bank": "ICICI Bank is an Indian private sector bank offering banking and financial services.",
    "Indian Oil Corporation Limited (IOCL)": "Indian Oil is India's national oil company; fuel, LPG, and petrol station network.",
    "Indian Railway Catering and Tourism Corporation (IRCTC)": "IRCTC handles online ticketing, catering and tourism for Indian Railways.",
    "National Informatics Centre (NIC)": "NIC is a central government agency that hosts many Indian government websites and portals.",
    "Punjab National Bank": "Punjab National Bank (PNB) is a public sector bank in India.",
    "State Bank of India (SBI)": "SBI is the largest public sector bank in India, offering banking services nationwide."
}

CSE_KEYWORDS = {
    "Airtel": ["airtel", "airtel.in", "airtelpayments", "airtelbank"],
    "Bank of Baroda": ["bankofbaroda", "bank of baroda", "bob"],
    "Civil Registration System, MHA": ["civil registration", "crs", "mha", "birth certificate", "death certificate"],
    "HDFC Bank": ["hdfc", "hdfcbank", "hdfc bank"],
    "ICICI Bank": ["icicibank", "icici bank"],
    "Indian Oil Corporation Limited (IOCL)": ["indianoil", "iocl", "indian oil", "petrol", "lpg"],
    "Indian Railway Catering and Tourism Corporation (IRCTC)": ["irctc", "railway ticket", "indianrail", "train ticket"],
    "National Informatics Centre (NIC)": ["nic.in", "national informatics centre"],
    "Punjab National Bank": ["pnb", "punjab national bank"],
    "State Bank of India (SBI)": ["sbi", "state bank of india", "onlinesbi"]
}


# ========== HELPERS ==========
def normalize_candidates(raw):
    raw = str(raw).strip()
    if raw == '':
        return []
    candidates = []
    if re.match(r'^https?://', raw, re.I):
        candidates.append(raw)
    else:
        candidates.extend([
            f'https://{raw}',
            f'http://{raw}',
            f'https://www.{raw}',
            f'http://www.{raw}'
        ])
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def fetch_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        return {"ok": True, "status_code": resp.status_code, "url": resp.url, "text": resp.text}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": str(e), "status_code": None, "url": url, "text": ""}


def try_fetch_variants(raw_domain):
    for c in normalize_candidates(raw_domain):
        for _ in range(RETRY_ATTEMPTS):
            res = fetch_url(c)
            if res["ok"]:
                return res
            time.sleep(0.5)
    return {"ok": False, "error": "unreachable", "status_code": None, "url": None, "text": ""}


def extract_visible_text(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text)
    return text[:15000]


def is_error_page(text):
    if not text:
        return True
    lower = text.lower()
    error_keywords = ["404", "not found", "error", "forbidden", "access denied", "timed out"]
    return any(k in lower for k in error_keywords)


def keyword_context_match(text):
    text_low = text.lower()
    # Strict NIC matching — must be in govt context
    if "nic.in" in text_low or "national informatics centre" in text_low:
        if any(g in text_low for g in ["gov.in", "ministry", "government", "portal", "india"]):
            return "National Informatics Centre (NIC)", 1.0, "keyword: nic.in with gov context"
        else:
            return None, 0.0, "nic found but no govt context"

    for cse, keys in CSE_KEYWORDS.items():
        for k in keys:
            if re.search(r'\b' + re.escape(k) + r'\b', text_low):
                return cse, 1.0, f"keyword: {k}"

    if "bank" in text_low:
        for bank, label in {
            "hdfc": "HDFC Bank",
            "icici": "ICICI Bank",
            "sbi": "State Bank of India (SBI)",
            "bank of baroda": "Bank of Baroda",
            "punjab national": "Punjab National Bank",
            "pnb": "Punjab National Bank"
        }.items():
            if bank in text_low:
                return label, 0.95, f"keyword: {bank} bank"
        return "Bank (generic)", 0.6, "generic bank keyword"
    return None, 0.0, None


# ========== Embedding-based Context Classifier ==========
class EmbeddingContextClassifier:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cse_names = list(CSE_LIST.keys())
        self.cse_texts = [CSE_LIST[k] for k in self.cse_names]
        self.cse_embeddings = self.model.encode(self.cse_texts, convert_to_tensor=True, show_progress_bar=False)

    def classify_text(self, text):
        if not text or text.strip() == "":
            return None, 0.0, "empty text"

        kw_match, kw_score, reason = keyword_context_match(text)
        if kw_match:
            return kw_match, kw_score, reason

        emb = self.model.encode(text, convert_to_tensor=True)
        hits = util.semantic_search(emb, self.cse_embeddings, top_k=1)
        if hits:
            best = hits[0][0]
            score = float(best['score'])
            label = self.cse_names[best['corpus_id']]
            if score < CONFIDENCE_THRESHOLD:
                return LOW_SCORE_LABEL, score, f"low embedding score ({score:.2f})"
            return label, score, f"embedding match ({label}) score={score:.2f}"
        return LOW_SCORE_LABEL, 0.0, "no embedding match"


# ========== MAIN LOGIC ==========
def process_row(raw_url, embed_classifier=None):
    ext = tldextract.extract(str(raw_url))
    domain_label = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    fetch_res = try_fetch_variants(raw_url)
    if not fetch_res.get("ok"):
        return {
            "Domain": domain_label,
            "RawInput": raw_url,
            "Fetched": False,
            "HTTP_Code": fetch_res.get("status_code"),
            "FinalURL": fetch_res.get("url"),
            "Error": fetch_res.get("error"),
            "Snippet": "",
            "Context_Label": LOW_SCORE_LABEL,
            "Context_Score": 0.0,
            "Reason": "fetch_failed"
        }

    text = extract_visible_text(fetch_res.get("text", ""))
    if is_error_page(text):
        return {
            "Domain": domain_label,
            "RawInput": raw_url,
            "Fetched": True,
            "HTTP_Code": fetch_res.get("status_code"),
            "FinalURL": fetch_res.get("url"),
            "Error": "",
            "Snippet": text[:800],
            "Context_Label": LOW_SCORE_LABEL,
            "Context_Score": 0.0,
            "Reason": "error/404 page"
        }

    cse_label, cse_score, reason = embed_classifier.classify_text(text) if embed_classifier else (LOW_SCORE_LABEL, 0.0, "no classifier")

    # Fuzzy fallback for borderline cases
    if cse_score < CONFIDENCE_THRESHOLD and text:
        best_name, best_ratio = None, 0
        for cname, desc in CSE_LIST.items():
            ratio = max(fuzz.partial_ratio(cname.lower(), text.lower()), fuzz.partial_ratio(desc.lower(), text.lower()))
            if ratio > best_ratio:
                best_name, best_ratio = cname, ratio
        if best_ratio > 70:
            cse_label, cse_score, reason = best_name, best_ratio / 100.0, f"fuzzy match {best_ratio}%"
        else:
            cse_label, cse_score, reason = LOW_SCORE_LABEL, 0.0, f"no reliable context (score={cse_score:.2f})"

    return {
        "Domain": domain_label,
        "RawInput": raw_url,
        "Fetched": True,
        "HTTP_Code": fetch_res.get("status_code"),
        "FinalURL": fetch_res.get("url"),
        "Error": "",
        "Snippet": text[:800],
        "Context_Label": cse_label,
        "Context_Score": round(cse_score, 3),
        "Reason": reason
    }


def main():
    args = parse_args()
    csv_path = args.csv
    output_path = args.output
    print("[INFO] Loading input CSV...")
    df = pd.read_csv(csv_path, header=None)
    raw_urls = df.iloc[:, 0].dropna().astype(str).tolist()

    embed_classifier = None
    if CONTEXT_METHOD == 'embedding':
        embed_classifier = EmbeddingContextClassifier(EMBEDDING_MODEL_NAME)

    print(f"[INFO] Processing {len(raw_urls)} URLs with {MAX_WORKERS} workers...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(process_row, url, embed_classifier): url for url in raw_urls}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Overall progress"):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({
                    "Domain": futures[fut],
                    "RawInput": futures[fut],
                    "Fetched": False,
                    "HTTP_Code": None,
                    "FinalURL": None,
                    "Error": str(e),
                    "Snippet": "",
                    "Context_Label": LOW_SCORE_LABEL,
                    "Context_Score": 0.0,
                    "Reason": "exception"
                })

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"[INFO] Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
