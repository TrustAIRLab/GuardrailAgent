import argparse
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Tuple, Iterable

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from model_runner import run_model_jd_agent

def _default_workers():
    try:
        return int(os.getenv("SLURM_CPUS_PER_TASK", "0")) or min(4, cpu_count())
    except Exception:
        return min(4, cpu_count())

def _load_items(path: str, limit: int | None) -> list[Dict[str, Any]]:
    p = Path(path)
    items: list[Dict[str, Any]] = []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if limit is not None:
        data = data[:limit]
    items = list(data)
    return items


def _worker(args: Tuple[int, Dict[str, Any]]) -> None:
    idx, item = args
    text = item.get("text") or item.get("prompt", "") 
    response = item.get("response", "")
    sample_id = item.get("id", idx)


    print(f"[WORKER] Processing #{idx} from {item.get('source')}")
    try:
        run_model_jd_agent(sample_id, text, response)
    except Exception as e:
        print(f"[ERROR] Item #{idx} failed: {e}")


def main():
    start_time = time.perf_counter()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON file")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N items")
    ap.add_argument("--workers", type=int, default=_default_workers(), help="Number of parallel workers")

    args = ap.parse_args()

    items = _load_items(args.input, args.limit)
    pairs: Iterable[Tuple[int, Dict[str, Any]]] = list(enumerate(items))

    total_items = len(items)
    print(f"[INFO] Loaded {total_items} items from {args.input}. Running with {args.workers} workers…")

    with Pool(processes=args.workers) as pool:
        for _ in pool.imap_unordered(_worker, pairs, chunksize=2):
            pass

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"[INFO] Done. Processed {total_items} items.")
    print(
        f"[TIME] Finished {total_items} items in "
        f"{elapsed:.2f} seconds ({elapsed/60:.2f} minutes, {elapsed/3600:.2f} hours)"
    )

if __name__ == "__main__":
    main()
