import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from P2O3_analysis.refinement_pipeline.refinement_pipeline import resolve_material

# --- Fixed configuration ---
req_files_prefix = ("Odash3", "Pdash2", "O3", "P2", "Na2CO3", "NaNO3")
pure_phase_prefixes = ("Odash3", "Pdash2", "O3", "P2")
excess_elements = {'O', 'Na'}

decimal_places = 6
threshold_for_impurity_significance = 0.02

xrd_dir = "xrd_data"
ref_dir = "ref_data"
exp_list_path = "Master List Experimental.csv"

thresholds = (2.1, 3)

MAX_WORKERS = 8

RESULTS_PATH = "batch_results.csv"
FIELDNAMES = [
    "uid", "original_formula", "adjusted_formula", "impurity_percentage",
    "p2_o3_ratio", "rwp", "flagged", "background_removal",
    "threshold_for_multi_val", "dedupe_by_spacegroup", "error",
]

# --- Per-worker state, set once by init_worker in each worker process ---
_worker_state = {}


def init_worker(exp_list_path, xrd_dir, ref_dir, req_files_prefix,
                 pure_phase_prefixes, excess_elements,
                 threshold_for_impurity_significance, decimal_places, thresholds):
    """Runs once per worker process, not once per task."""
    _worker_state["df_exp_list"] = pd.read_csv(exp_list_path)
    _worker_state["xrd_dir"] = xrd_dir
    _worker_state["ref_dir"] = ref_dir
    _worker_state["req_files_prefix"] = req_files_prefix
    _worker_state["pure_phase_prefixes"] = pure_phase_prefixes
    _worker_state["excess_elements"] = excess_elements
    _worker_state["threshold_for_impurity_significance"] = threshold_for_impurity_significance
    _worker_state["decimal_places"] = decimal_places
    _worker_state["thresholds"] = thresholds


def worker_task(uid):
    """Runs in a worker process, once per material. Only takes `uid` so
    nothing bulky needs to be pickled and sent per-task."""
    s = _worker_state
    return resolve_material(
        s["df_exp_list"], uid, s["xrd_dir"], s["ref_dir"],
        s["req_files_prefix"], s["pure_phase_prefixes"], s["excess_elements"],
        s["threshold_for_impurity_significance"], s["decimal_places"], s["thresholds"],
    )


def load_completed_uids(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return {row["uid"] for row in reader}


def append_result(result, path):
    file_exists = os.path.exists(path)
    row = dict(result)
    row["flagged"] = json.dumps(row.get("flagged")) if row.get("flagged") is not None else ""

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def print_progress(result, index, total):
    flagged = result.get("flagged")
    if flagged:
        flag_str = ",".join(flagged.keys())
    elif flagged == {}:
        flag_str = "none"
    else:
        flag_str = "N/A"

    print(
        f"[{index}/{total}] {result['uid']:<10} "
        f"flagged={flag_str:<15} "
        f"rwp={result.get('rwp')} "
        f"dedupe={result.get('dedupe_by_spacegroup')} "
        f"bg={result.get('background_removal')} "
        f"thresh={result.get('threshold_for_multi_val')}",
        flush=True,
    )


def main():
    df_exp_list = pd.read_csv(exp_list_path)
    all_uids = df_exp_list["UID"].tolist()
    total = len(all_uids)

    already_done = load_completed_uids(RESULTS_PATH)
    remaining = [uid for uid in all_uids if uid not in already_done]

    if already_done:
        print(f"Resuming: {len(already_done)} of {total} already completed.")
    print(f"{len(remaining)} remaining, running with {MAX_WORKERS} workers.")

    completed_count = len(already_done)

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_worker,
        initargs=(exp_list_path, xrd_dir, ref_dir, req_files_prefix,
                  pure_phase_prefixes, excess_elements,
                  threshold_for_impurity_significance, decimal_places, thresholds),
    ) as executor:
        futures = {executor.submit(worker_task, uid): uid for uid in remaining}

        for future in as_completed(futures):
            uid = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "uid": uid,
                    "original_formula": None,
                    "adjusted_formula": None,
                    "impurity_percentage": None,
                    "p2_o3_ratio": None,
                    "rwp": None,
                    "flagged": None,
                    "background_removal": None,
                    "threshold_for_multi_val": None,
                    "dedupe_by_spacegroup": None,
                    "error": f"worker process raised: {e}",
                }

            append_result(result, RESULTS_PATH)
            completed_count += 1
            print_progress(result, completed_count, total)

    print("Batch run complete.")


if __name__ == "__main__":
    main()