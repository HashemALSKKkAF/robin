import json
import threading
import os
from pathlib import Path
from typing import List, Dict
from html_txt import HtmlToTxt

# Import Loader class directly
from loader import Loader

SEEDS_PATH = str(Path.home() / "Obscura/data/seeds.json")


class AutomatedLoader:
    # class-level locks (shared if multiple instances exist)
    stack_lock = threading.Lock()
    json_lock = threading.Lock()

    def __init__(self):
        # working stack of seed dicts (only the frozen selection for this run)
        self.stack: List[Dict] = []
        # URLs that finished processing in this run
        self.completed: List[str] = []
        self.completed_lock = threading.Lock()
        # URLs that were assigned to workers (prevents duplicates)
        self.assigned = set()

    # ---------------------------------------------------------
    # Load JSON seeds (no writes)
    # ---------------------------------------------------------
    def load_seeds(self) -> List[Dict]:
        with open(SEEDS_PATH, "r") as f:
            return json.load(f)["seeds"]

    # ---------------------------------------------------------
    # Save seeds.json (ONLY intended real updates)
    # ---------------------------------------------------------
    def save_seeds(self, seeds: List[Dict]):
        # single point of write; protected
        with self.json_lock:
            with open(SEEDS_PATH, "w") as f:
                json.dump({"seeds": seeds}, f, indent=2)

    # ---------------------------------------------------------
    # Pick up to 5 entries: crawled=True AND loaded=False
    # ---------------------------------------------------------
    def pick_five_unloaded(self) -> List[Dict]:
        seeds = self.load_seeds()
        selected = []
        seen_urls = set()

        for seed in seeds:
            url = seed.get("url")
            if url in seen_urls:
                continue
            if seed.get("crawled") is True and seed.get("loaded") is False:
                selected.append(seed)
                seen_urls.add(url)
            if len(selected) == 5:
                break

        return selected

    # ---------------------------------------------------------
    # Coordinator: Extracts seeds and pushes into stack (frozen)
    # ---------------------------------------------------------
    def coordinator_thread(self):
        selected = self.pick_five_unloaded()
        # ensure dedupe and cap to 5 (defensive)
        unique_selected = []
        urls_seen = set()
        for s in selected:
            url = s.get("url")
            if url and url not in urls_seen:
                unique_selected.append(s.copy())  # shallow copy so workers can't mutate shared dict
                urls_seen.add(url)
            if len(unique_selected) == 5:
                break

        print(f"[Coordinator] Selected {len(unique_selected)} seeds to load.")

        # freeze the selection for this run and mark assigned
        with self.stack_lock:
            # clear old stack (avoid leftovers if instance persists)
            self.stack = unique_selected
            # populate assigned set so no duplicates will be processed
            for s in unique_selected:
                if s.get("url"):
                    self.assigned.add(s["url"])

    # ---------------------------------------------------------
    # Worker thread: pops safely and calls Loader directly
    # ---------------------------------------------------------
    def worker_thread(self):
        while True:
            with self.stack_lock:
                if not self.stack:
                    # nothing left for this worker
                    return
                seed = self.stack.pop()

            url = seed.get("url")
            if not url:
                # skip malformed
                continue

            # Defensive: if already completed by another worker just skip
            with self.completed_lock:
                if url in self.completed:
                    continue

            print(f"[Worker {threading.get_ident()}] Loading {url}")

            # Run loader (may have side effects elsewhere; we treat it as black box)
            try:
                loader = Loader(url, seed.get("hash"))
                exit_code = loader.run()
            except Exception as e:
                print(f"[Worker {threading.get_ident()}] Loader error for {url}: {e}")
                exit_code = -1

            print(f"[Worker {threading.get_ident()}] Done {url} (exit={exit_code})")

            # -----------------------------------------
            # HTML → TXT integration
            # -----------------------------------------
            try:
                print(f"[Worker {threading.get_ident()}] Running html_txt on hash={seed.get('hash')}")
                HtmlToTxt.run(seed.get("hash"))
            except Exception as e:
                print(f"[Worker {threading.get_ident()}] ERROR in html_txt for {url}: {e}")

            # Mark completed (thread-safe)
            with self.completed_lock:
                if url not in self.completed:
                    self.completed.append(url)

    # ---------------------------------------------------------
    # Update JSON only once after all workers finish
    # ---------------------------------------------------------
    def update_json(self):
        # Load the latest seeds.json (it may have been touched by Loader,
        # but we'll set loaded=True for the URLs we actually completed this run)
        seeds = self.load_seeds()

        with self.completed_lock:
            done = set(self.completed)

        modified = False
        for seed in seeds:
            if seed.get("url") in done and seed.get("loaded") is not True:
                seed["loaded"] = True
                modified = True

        if modified:
            self.save_seeds(seeds)
            print("[JSON] Updated loaded=true for completed seeds.")
        else:
            print("[JSON] No changes needed to seeds.json.")

    # ---------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------
    @staticmethod
    def run():
        auto = AutomatedLoader()

        # Single coordinator thread (freezes selection for this run)
        coord = threading.Thread(target=auto.coordinator_thread)
        coord.start()
        coord.join()

        # Worker threads: create up to min(5, len(stack)) workers
        with auto.stack_lock:
            worker_count = min(5, len(auto.stack))
        if worker_count == 0:
            print("[AutomatedLoader] No seeds to process.")
            return

        workers = []
        for _ in range(worker_count):
            t = threading.Thread(target=auto.worker_thread)
            t.start()
            workers.append(t)

        for t in workers:
            t.join()

        # Commit final JSON state
        auto.update_json()

        print("[AutomatedLoader] All sites processed.")


if __name__ == "__main__":
    AutomatedLoader.run()
