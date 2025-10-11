import pandas as pd
import requests
import validators
import argparse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/140.0.0.0 Safari/537.36"
}

def check_url(url: str) -> str:
    url = url.strip()
    try:
        response = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        if response.status_code < 400:
            return "found"
        else:
            return "not found"
    except requests.RequestException:
        return "not found"

def crawl_urls(csv_path: str, url_column: str = "domain", output_csv: str = "results.csv"):
    df = pd.read_csv(csv_path)

    if url_column not in df.columns:
        raise ValueError(f"CSV must contain a column named '{url_column}'")

    results = []
    for url in df[url_column]:
        url = str(url).strip()
        if not validators.url(url):
            # print(f"{url} -> invalid, skipping")
            results.append("not found")
            continue
        status = check_url(str(url).strip())
        print(f"{url} -> {status}")
        results.append(status)

    df["status"] = results
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Check status of URLs in a CSV file.")
    parser.add_argument('--gen_url_path', type=str, required=True, help="Path to input CSV")
    parser.add_argument('--output_csv', type=str, default="results.csv", help="Path to save results")
    args = parser.parse_args()

    crawl_urls(args.gen_url_path, output_csv=args.output_csv)

if __name__ == "__main__":
    main()
