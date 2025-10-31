import subprocess
import os

brands = [
    'airtel.in', 'bankofbaroda.in', 'dc.crsorgi.gov.in', 'hdfcbank.com',
    'hdfcergo.com', 'hdfclife.com', 'icicibank.com', 'icicidirect.com',
    'icicilombard.com', 'iciciprulife.com', 'iocl.com', 'irctc.co.in',
    'ncrb.gov.in', 'email.gov.in', 'kavach.mail.gov.in', 'accounts.mgovcloud.in',
    'nic.gov.in', 'pnbindia.in', 'sbicard.com', 'sbilife.co.in', 'sbi.co.in'
]

SAVE_DIR = "src\saved_models"
GENERATED_DIR = "data\generated"
CRAWLER_SCRIPT = "src.crawler.crawler"
RESULTS_DIR = "results\lstm_generator"
CSV_PATH = "results\domains_found.csv"

os.makedirs(GENERATED_DIR, exist_ok=True)

def run_command(command):
    """Utility to run shell commands cross-platform"""
    try:
        print(f"\n>>> Running: {command}\n")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")

def main():
    for brand in brands:
        safe_brand = brand.replace(".", "_")  # e.g. airtel_in
        gen_csv = os.path.join(GENERATED_DIR, f"generated_{safe_brand}.csv")
        out_csv = os.path.join(RESULTS_DIR, f"result_{safe_brand}.csv")

        # Step 1: Run URL generator
        gen_cmd = (f"python -m src.data_pipeline.generate_urls --mode generate --save_dir {SAVE_DIR} --demo_brand {brand} --hidden 128 --temperature 1 --top_k 40 --top_p 0.95 --n_samples 200 --out_csv {gen_csv}")
        run_command(gen_cmd)

        # Step 2: Run crawler
        crawl_cmd = f"python -m {CRAWLER_SCRIPT} --gen_url_path {gen_csv} --output_csv {out_csv}"
        run_command(crawl_cmd)

        print(f"Finished processing {brand}: {gen_csv} → {out_csv}")

    inference_cmd = (f"python -m src.inference.inference_cnnbilstm --csv {CSV_PATH}")
    run_command(inference_cmd)

    print("Finished Classification of candidate urls with output file results/url_predictions.csv")

if __name__ == "__main__":
    main()
