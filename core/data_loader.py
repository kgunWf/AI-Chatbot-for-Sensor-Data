# core/data_loader.py
import os
import json
from typing import Iterable, Dict

from stdatalog_loader import iter_hsd_items
from feature_extraction import extract_features_from_bags
import pandas as pd


def load_hsd_items(root: str):
    # root can be your top-level dataset, e.g. "/Users/kgun/Downloads/Sensor_STWIN"
    for item in iter_hsd_items(root, only_active=True, verbose=True):
        yield item


def test_hsd_loader(root: str, limit: int = 10, verbose: bool = True) -> None:
    """
    Original quick scan: prints one line per sensor stream.
    """
    n = 0
    for item in iter_hsd_items(root, only_active=True, verbose=verbose):
        n += 1
        rows = len(item["data"]) if hasattr(item.get("data"), "__len__") else -1
        print(
            f"{n:04d} | {item['condition']} | {item['belt_status']} | "
            f"{item['sensor']} | {item['sensor_type']} | {item['rpm']} | rows={rows} | odr={item.get('odr')}"
        )
        if n >= limit:
            break
    print(f"\nDone. Printed {min(n, limit)} of {n} loaded items.")


def test_hsd_loader_with_features(root: str, limit: int = 10, verbose: bool = True) -> pd.DataFrame:
    """
    New: also computes features for up to `limit` databags and returns a DataFrame.
    """
    bags = []
    n = 0
    for item in iter_hsd_items(root, only_active=True, verbose=verbose):
        n += 1
        rows = len(item["data"]) if hasattr(item.get("data"), "__len__") else -1
        print(
            f"{n:04d} | {item['condition']} | {item['belt_status']} | "
            f"{item['sensor']} | {item['sensor_type']} | {item['rpm']} | rows={rows} | odr={item.get('odr')}"
        )
        bags.append(item)
        if n >= limit:
            break

    # compute features
    feats_df = extract_features_from_bags(bags)
    # put a few identifying cols first
    id_cols = ["condition", "belt_status", "sensor", "rpm"]
    other_cols = [c for c in feats_df.columns if c not in id_cols]
    feats_df = feats_df[id_cols + other_cols]

    # quick peek
    if verbose:
        with pd.option_context("display.max_columns", 50, "display.width", 300):
            print("\nFeature preview (first 5 rows, first 50 cols):")
            print(feats_df.iloc[:5, :50])

    return feats_df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test HSD loader + features via core/data_loader.py")
    ap.add_argument("root", nargs="?", default=os.getenv("STAT_AI_DATA", ""), help="Dataset root path")
    ap.add_argument("--limit", type=int, default=10, help="Max items to process")
    ap.add_argument("--quiet", action="store_true", help="Less verbose")
    ap.add_argument("--features", action="store_true", help="Also compute features and print a preview")
    ap.add_argument("--save-features", type=str, default="", help="Path to save features as CSV (optional)")
    args = ap.parse_args()

    if not args.root:
        raise SystemExit("Provide a dataset root path or set STAT_AI_DATA")

    if args.features or args.save_features:
        feats_df = test_hsd_loader_with_features(args.root, limit=args.limit, verbose=(not args.quiet))
        if args.save_features:
            out_path = args.save_features
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            feats_df.to_csv(out_path, index=False)
            print(f"\n✅ Saved features to: {out_path}")
    else:
        test_hsd_loader(args.root, limit=args.limit, verbose=(not args.quiet))
