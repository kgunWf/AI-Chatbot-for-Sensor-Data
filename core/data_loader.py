# core/data_loader.py
import os
from pathlib import Path
from typing import Iterable, Dict, List

from stdatalog_loader import iter_hsd_items
from feature_extraction import extract_features_from_bag, extract_features_from_bags
from feature_analysis import analyze_features_from_bags

import pandas as pd


def test_hsd_loader(root: str, limit: int = 10, verbose: bool = True) -> None:
    """
    Original quick scan: prints one line per sensor stream.
    """
    n = 0
    bags = iter_hsd_items(root, only_active=True, verbose=verbose)
    for item in bags:
        n += 1
        rows = len(item["data"]) if hasattr(item.get("data"), "__len__") else -1
        print(
            f"{n:04d} | {item['sensor']} | {item['sensor_type']} | "
            f"rows={rows} | odr={item.get('odr')}"
        )
        if n >= limit:
            break
    print(f"\nDone. Printed {min(n, limit)} of {n} loaded items.")


def test_hsd_loader_with_features(root: str, limit: int = 10, fetch_all: bool = True, verbose: bool = False) -> list[dict]:
    """
    Extract features from up to `limit` databags and return as a list of feature dictionaries.
    """
    bags = []
    n = 0
    if not fetch_all:
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
    else: 
        bags = iter_hsd_items(root, only_active=True, verbose=verbose)
    feature_dicts = extract_features_from_bags(bags)

    if verbose:
        df_preview = pd.DataFrame(feature_dicts)
        id_cols = ["condition", "belt_status", "sensor", "rpm"]
        other_cols = [c for c in df_preview.columns if c not in id_cols]
        df_preview = df_preview[id_cols + other_cols]
        with pd.option_context("display.max_columns", 50, "display.width", 300):
            print("\nFeature preview (first 5 rows, first 50 cols):")
            print(df_preview.iloc[:5, :50])

    return feature_dicts


def test_feature_importance_analysis(feature_dicts: list[dict], limit: int = 10):
    """
    Analyzes feature importance per sensor type from a list of extracted feature dicts.
    """
    print(f"\n🔎 Running feature importance analysis across sensor types (top {limit} features each)...")
    results = analyze_features_from_bags(feature_dicts, top_k=limit)

    for sensor_type, result in results.items():
        print(f"\n📊 Sensor Type: {sensor_type}")
        print(f"✅ Accuracy: {result['accuracy']:.2f}")
        print("Top features (Random Forest):")
        print(result["rf_importances"].head(limit))
        print("Top features (ANOVA F-score):")
        print(result["anova_scores"].head(limit))


#Important: you can test features for a specific sensor type only, e.g., 'acc', 'temp', 'hum', 'mic'
#I used this during development to focus on one sensor at a time
def test_features_by_sensor_type(root: str, sensor_type: str, limit: int = 5) -> None:
    """
    Test and print features only for a specific sensor type (e.g., 'acc', 'temp', 'hum', 'mic').
    """
    count = 0
    for item in iter_hsd_items(root, only_active=True, verbose=False):
        if item["sensor_type"] != sensor_type:
            continue

        print(f"\n🔍 [{count+1}] Sensor: {item['sensor']} | Type: {sensor_type} | Rows: {len(item['data'])}")
        features = extract_features_from_bag(item)
        df = pd.DataFrame([features])

        # Visualize first few features
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df.T.head(20))  # Transpose to view vertically

        count += 1
        if count >= limit:
            break

    print(f"\n✅ Done. Displayed {count} {sensor_type} sensor features.")



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test HSD loader and feature analysis using core/data_loader.py")
    ap.add_argument("root", nargs="?", default=os.getenv("STAT_AI_DATA", ""), help="Dataset root path (default: $STAT_AI_DATA)")
    ap.add_argument("--limit", type=int, default=10, help="Max number of sensors to process")
    ap.add_argument("--quiet", action="store_true", help="Suppress detailed logs")
    ap.add_argument("--features", action="store_true", help="Extract and preview sensor features")
    ap.add_argument("--save-features", type=str, default="", help="Save extracted features to a CSV file (used with --features)")

    args = ap.parse_args()

    if not args.root:
        raise SystemExit("❌ Please provide a dataset root path or set the STAT_AI_DATA environment variable.")

    if args.features or args.save_features:
        feature_dicts = test_hsd_loader_with_features(args.root, limit=args.limit, verbose=(not args.quiet))

        if args.save_features:
            out_path = args.save_features
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            pd.DataFrame(feature_dicts).to_csv(out_path, index=False)
            print(f"\n✅ Saved extracted features to: {out_path}")

        test_feature_importance_analysis(feature_dicts, limit=20)

    else:
        test_hsd_loader(args.root, limit=args.limit, verbose=(not args.quiet))

