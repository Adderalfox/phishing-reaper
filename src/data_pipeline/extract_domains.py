import os
import pandas as pd

# ========== Config ==========
RESULTS_DIR = "results/lstm_generator"
OUTPUT_CSV = "results/found_domains.csv"

def main():
    all_found_domains = []

    # Iterate through all CSV files in results directory
    for file_name in os.listdir(RESULTS_DIR):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(RESULTS_DIR, file_name)
        print(f"Processing: {file_path}")

        try:
            df = pd.read_csv(file_path)

            # Normalize column names (lowercase, strip spaces)
            df.columns = [col.strip().lower() for col in df.columns]

            if "domain" not in df.columns or "status" not in df.columns:
                print(f"Skipping {file_name} — required columns missing.")
                continue

            # Filter rows where status == 'found'
            found_df = df[df["status"].str.lower() == "found"]

            if not found_df.empty:
                brand_name = file_name.replace("result_", "").replace(".csv", "")
                found_df["source_brand_file"] = brand_name  # track where it came from
                all_found_domains.append(found_df[["domain", "source_brand_file"]])

        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    # Combine and save results
    if all_found_domains:
        combined_df = pd.concat(all_found_domains, ignore_index=True)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        combined_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nCombined CSV saved: {OUTPUT_CSV}")
        print(f"Total found domains: {len(combined_df)}")
    else:
        print("\nNo 'found' domains detected in any CSV.")

if __name__ == "__main__":
    main()
