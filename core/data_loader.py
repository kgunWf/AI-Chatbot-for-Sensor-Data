# core/data_loader.py
import os
from pathlib import Path
from typing import Iterable, Dict, List
from unittest import result
import json
from stdatalog_loader import iter_hsd_items
from feature_extraction import extract_features_from_bag, extract_features_from_bags, prepare_combined_feature_dataframe
from feature_analysis import analyze_global_features
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
    cleaned_df = prepare_combined_feature_dataframe(feature_dicts)

    if verbose:
        with pd.option_context("display.max_columns", 50, "display.width", 300):
            print("\nFeature preview (first 5 rows, first 50 cols):")
            print(cleaned_df.iloc[:5, :50])

    return feature_dicts, cleaned_df


def test_feature_importance_analysis(df: pd.DataFrame, limit: int = None, sensor_subset: list[str] = None) -> None:
    """
    Runs GLOBAL feature importance analysis across ALL sensors.
    Produces:
        - global RF feature importances
        - global ANOVA F-scores
        - top K features
    """

    print(f"\n🔎 Running GLOBAL feature importance analysis (top {limit} features)...")


    print("\n🧹 Cleaned & Imputed Feature Matrix Preview:")
    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print(df.head())
    if limit is not None:
        # 1. Manual mode: user provides top_k
        result = analyze_global_features(df, sensor_subset=sensor_subset, top_k=limit, do_plots=True)
        print(f"\nTop {limit} features (manual mode):")
        print(result["top_features"])
    # 2. Run global analysis-in auto mode
    result = analyze_global_features(df, sensor_subset=sensor_subset, do_plots=True)
    print("Auto-selected K =", result["selected_k"])
    print("Best accuracy =", result["best_accuracy"])
    print(result["top_features"])


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

def load_raw_bags(root: str, limit: int | None = None, verbose: bool = False) -> list[dict]:
    bags = []
    for i, item in enumerate(iter_hsd_items(root, only_active=True, verbose=verbose), start=1):
        bags.append(item)
        if limit is not None and i >= limit:
            break
    return bags

def filter_bags(
    bags: list[dict],
    sensor_type: str | None = None,
    sensor: str | None = None,
    belt_status: str | None = None,
    condition: str | None = None,
) -> list[dict]:
    """
    Return only the bags that match the given criteria.
    Any argument left as None is ignored.
    """
    out = []
    for b in bags:
        if sensor_type is not None and b["sensor_type"] != sensor_type:
            continue #keep only the bags that match the given sensor type
        if sensor is not None and b["sensor"] != sensor:
            continue #keep only the bags that match the given sensor name (e.g. IIS3DWB etc.)
        #burda ko, low, 2mm tek tek de filtrelenebilir
        if belt_status is not None and str(b["belt_status"]) != belt_status:
            continue #keep only the bags that match the given belt status(e.g. OK, KO_HIGH_2mm, KO_LOW_2mm,)#
        if condition is not None and str(b["condition"]) != condition:
            continue #keep only the bags that match the given condition(e.g. vel-fissa, no-load-cycles)
        out.append(b)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test HSD loader and feature analysis using core/data_loader.py")
    ap.add_argument("root", nargs="?", default=os.getenv("STAT_AI_DATA", ""), help="Dataset root path (default: $STAT_AI_DATA)")
    ap.add_argument("--limit", type=int, default=10, help="Max number of sensors to process")
    ap.add_argument("--quiet", action="store_true", help="Suppress detailed logs")
    ap.add_argument("--features", action="store_true", help="Extract and preview sensor features")
    ap.add_argument("--save_features", type=str, default="", help="Save extracted features to a CSV file (used with --features)")
    ap.add_argument("--load_features", type=str, default="", help="Load features from a CSV file instead of extracting")
    args = ap.parse_args()

    if not args.root:
        raise SystemExit("❌ Please provide a dataset root path or set the STAT_AI_DATA environment variable.")

    if args.load_features:
        print(f"📥 Loading features from: {args.load_features}")
        cleaned_df_path = args.load_features + "/cleaned_df.csv"
        cleaned_df = pd.read_csv(cleaned_df_path)

        feature_dicts_path = args.load_features + "/dictionary.json"
        with open(feature_dicts_path, "r") as f:
            feature_dicts = json.load(f)

        print(cleaned_df["belt_status"].value_counts())
       # print(cleaned_df.groupby("path")["sensor_type"].apply(list).head(20))


        if cleaned_df is None:
            df = prepare_combined_feature_dataframe(feature_dicts)
        else:
            df = cleaned_df

        #sensor subset defines the most important sensors to consider during feature analysis, temp, hum, prs are not very relevant, and hum doesn't have data
        sensor_subset = ["acc", "mic", "mag"]
        test_feature_importance_analysis(cleaned_df, sensor_subset=sensor_subset)
        raise SystemExit(0)
    

    if args.features or args.save_features:

        feature_dicts, cleaned_df = test_hsd_loader_with_features(
            args.root,
            limit=args.limit,
            verbose=(not args.quiet)
        )

        # If user specifies a directory in args.save_features
        if args.save_features:
            out_dir = args.save_features
            os.makedirs(out_dir, exist_ok=True)

            # ---- Save dictionary.json ----
            dict_path = os.path.join(out_dir, "dictionary.json")
            with open(dict_path, "w") as f:
                json.dump(feature_dicts, f, indent=4)
            print(f"✅ Saved features_dict to: {dict_path}")

            # ---- Save cleaned_df.csv ----
            df_path = os.path.join(out_dir, "cleaned_df.csv")
            cleaned_df.to_csv(df_path, index=False)
            print(f"✅ Saved cleaned_df to: {df_path}")

        test_feature_importance_analysis(cleaned_df, limit=20)


    else:
        test_hsd_loader(args.root, limit=args.limit, verbose=(not args.quiet))

