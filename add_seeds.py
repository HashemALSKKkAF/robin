import json
import os
import hashlib

SEEDS_FILE = os.path.expanduser("~/Obscura/data/seeds.json")

def load_seeds():
    """Load existing seeds.json file."""
    if not os.path.exists(SEEDS_FILE):
        print(f"[!] seeds.json not found at: {SEEDS_FILE}")
        return {"seeds": []}

    with open(SEEDS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_seeds(data):
    """Save updates back to seeds.json."""
    with open(SEEDS_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        print("[+] Seed added successfully.")


def sha256_hash(value: str) -> str:
    """Generate SHA-256 hash of a string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_seed_entry(url: str, name: str) -> dict:
    """Create a new seed entry following your exact JSON structure."""
    if url.endswith("/"):
        host = url
    else:
        host = url + "/"

    return {
        "host": host,
        "url": url,
        "hash": sha256_hash(url),
        "name": name if name.strip() else "unknown",
        "status_code": None,
        "crawled": False,
        "loaded": False,
        "c_matched": False,
        "l_matched": False
    }


def main():
    print("=== Add New Seed Entry ===\n")

    url = input("Enter the URL: ").strip()
    if not url:
        print("[!] No URL provided. Exiting.")
        return

    name = input("Enter a name (press Enter for 'unknown'): ").strip()

    seeds_data = load_seeds()

    new_entry = create_seed_entry(url, name)
    seeds_data["seeds"].append(new_entry)

    save_seeds(seeds_data)

    print("\n[+] New seed entry created:")
    print(json.dumps(new_entry, indent=2))


if __name__ == "__main__":
    main()
