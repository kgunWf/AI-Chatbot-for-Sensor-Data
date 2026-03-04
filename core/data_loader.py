# core/data_loader.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stdatalog_loader import iter_hsd_items


DEFAULT_OUTPUT_DIR = "output_dir/processed"


@dataclass
class BagFilters:
    sensor_type: str | None = None
    sensor: str | None = None
    belt_status: str | None = None
    condition: str | None = None
    rpm: str | None = None


def fetch_bags(
    root: str | Path,
    limit: int | None = None,
    only_active: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Load HSD bags from disk with optional row limit."""
    bags: list[dict[str, Any]] = []
    for index, item in enumerate(
        iter_hsd_items(root, only_active=only_active, verbose=verbose), start=1
    ):
        bags.append(item)
        if limit is not None and index >= limit:
            break
    return bags


def filter_bags(
    bags: list[dict[str, Any]],
    sensor_type: str | None = None,
    sensor: str | None = None,
    belt_status: str | None = None,
    condition: str | None = None,
    rpm: str | None = None,
) -> list[dict[str, Any]]:
    """Return bags matching provided filters. None means no filter."""
    filters = BagFilters(
        sensor_type=sensor_type,
        sensor=sensor,
        belt_status=belt_status,
        condition=condition,
        rpm=rpm,
    )
    out: list[dict[str, Any]] = []
    for bag in bags:
        if filters.sensor_type and bag.get("sensor_type") != filters.sensor_type:
            continue
        if filters.sensor and bag.get("sensor") != filters.sensor:
            continue
        if filters.belt_status and str(bag.get("belt_status")) != filters.belt_status:
            continue
        if filters.condition and str(bag.get("condition")) != filters.condition:
            continue
        if filters.rpm and str(bag.get("rpm")) != filters.rpm:
            continue
        out.append(bag)
    return out


def group_by_sensor_name(
    bags: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for bag in bags:
        grouped.setdefault(str(bag.get("sensor", "unknown")), []).append(bag)
    return grouped


def generate_features(
    bags: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Create feature dictionaries and cleaned feature dataframe."""
    from feature_extraction import (
        extract_features_from_bags,
        prepare_combined_feature_dataframe,
    )

    feature_dicts = extract_features_from_bags(bags)
    cleaned_df = prepare_combined_feature_dataframe(feature_dicts)
    return feature_dicts, cleaned_df


def save_feature_artifacts(
    output_dir: str | Path,
    feature_dicts: list[dict[str, Any]],
    cleaned_df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Save feature artifacts in a stable layout for the LLM interface:
    - dictionary.json
    - cleaned_df.csv
    - manifest.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dictionary_path = output_path / "dictionary.json"
    dataframe_path = output_path / "cleaned_df.csv"
    manifest_path = output_path / "manifest.json"

    with dictionary_path.open("w", encoding="utf-8") as file:
        json.dump(feature_dicts, file, indent=2, ensure_ascii=True)
    cleaned_df.to_csv(dataframe_path, index=False)

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_bags": len(feature_dicts),
        "num_rows": int(cleaned_df.shape[0]),
        "num_columns": int(cleaned_df.shape[1]),
        "files": {
            "dictionary": dictionary_path.name,
            "dataframe": dataframe_path.name,
        },
    }
    if metadata:
        manifest.update(metadata)

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=True)

    return {
        "output_dir": str(output_path.resolve()),
        "dictionary": str(dictionary_path.resolve()),
        "cleaned_df": str(dataframe_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }


def load_feature_artifacts(
    output_dir: str | Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """
    Load feature artifacts from either:
    - an artifacts directory containing cleaned_df.csv (+ optional dictionary/manifest)
    - a direct path to cleaned_df.csv
    """
    input_path = Path(output_dir)

    if input_path.is_file():
        if input_path.name != "cleaned_df.csv":
            raise FileNotFoundError(
                "When --load points to a file, it must be 'cleaned_df.csv'."
            )
        output_path = input_path.parent
        dataframe_path = input_path
    else:
        output_path = input_path
        dataframe_path = output_path / "cleaned_df.csv"

    dictionary_path = output_path / "dictionary.json"
    manifest_path = output_path / "manifest.json"

    if not dataframe_path.exists():
        raise FileNotFoundError(f"Missing cleaned dataframe file: {dataframe_path}")

    cleaned_df = pd.read_csv(dataframe_path)

    if dictionary_path.exists():
        with dictionary_path.open("r", encoding="utf-8") as file:
            feature_dicts = json.load(file)
    else:
        feature_dicts = []

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    else:
        manifest = {}

    return feature_dicts, cleaned_df, manifest


def data_preprocessing(
    root: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    only_active: bool = True,
    verbose: bool = False,
    filters: BagFilters | None = None,
) -> dict[str, str]:
    """
    End-to-end pipeline:
    1) fetch raw bags
    2) apply optional filters
    3) generate features
    4) save artifacts to output_dir
    """
    bags = fetch_bags(root=root, limit=limit, only_active=only_active, verbose=verbose)

    if filters is not None:
        bags = filter_bags(
            bags,
            sensor_type=filters.sensor_type,
            sensor=filters.sensor,
            belt_status=filters.belt_status,
            condition=filters.condition,
            rpm=filters.rpm,
        )

    if not bags:
        raise ValueError("No bags found after loading/filtering; cannot generate features.")

    feature_dicts, cleaned_df = generate_features(bags)
    metadata = {
        "source_root": str(Path(root).resolve()),
        "only_active": bool(only_active),
        "filters": filters.__dict__ if filters else {},
    }
    return save_feature_artifacts(
        output_dir=output_dir,
        feature_dicts=feature_dicts,
        cleaned_df=cleaned_df,
        metadata=metadata,
    )


def _print_preview(cleaned_df: pd.DataFrame, rows: int = 5) -> None:
    print(f"\nFeature dataframe shape: {cleaned_df.shape}")
    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print(cleaned_df.head(rows))


def filter_feature_dataframe(
    cleaned_df: pd.DataFrame,
    filters: BagFilters | None = None,
) -> pd.DataFrame:
    """Filter an already-generated cleaned feature dataframe by metadata columns."""
    if filters is None:
        return cleaned_df

    out = cleaned_df.copy()
    if filters.sensor_type is not None and "sensor_type" in out.columns:
        out = out[out["sensor_type"].astype(str) == filters.sensor_type]
    if filters.sensor is not None and "sensor" in out.columns:
        out = out[out["sensor"].astype(str) == filters.sensor]
    if filters.belt_status is not None and "belt_status" in out.columns:
        out = out[out["belt_status"].astype(str) == filters.belt_status]
    if filters.condition is not None and "condition" in out.columns:
        out = out[out["condition"].astype(str) == filters.condition]
    if filters.rpm is not None and "rpm" in out.columns:
        out = out[out["rpm"].astype(str) == filters.rpm]
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="User-friendly data loading + feature generation pipeline for LLM analysis."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=os.getenv("STAT_AI_DATA", ""),
        help="Dataset root path (default: $STAT_AI_DATA).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output folder for artifacts (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max bags to load.")
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive sensors (default behavior uses only active sensors).",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce loader verbosity.")
    parser.add_argument("--preview", action="store_true", help="Print a dataframe preview.")
    parser.add_argument("--load", type=str, default="", help="Load existing artifacts directory.")

    parser.add_argument("--sensor-type", type=str, default=None, help="Filter by sensor_type.")
    parser.add_argument("--sensor", type=str, default=None, help="Filter by sensor name.")
    parser.add_argument("--belt-status", type=str, default=None, help="Filter by belt_status.")
    parser.add_argument("--condition", type=str, default=None, help="Filter by condition.")
    parser.add_argument("--rpm", type=str, default=None, help="Filter by rpm.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    filters = BagFilters(
        sensor_type=args.sensor_type,
        sensor=args.sensor,
        belt_status=args.belt_status,
        condition=args.condition,
        rpm=args.rpm,
    )
    has_filters = any(value is not None for value in filters.__dict__.values())

    root_path = Path(args.root) if args.root else None
    root_looks_like_artifacts = bool(
        root_path
        and root_path.exists()
        and (
            (root_path.is_file() and root_path.name == "cleaned_df.csv")
            or (root_path.is_dir() and (root_path / "cleaned_df.csv").exists())
        )
    )
    load_input = args.load or (args.root if root_looks_like_artifacts else "")

    if load_input:
        feature_dicts, cleaned_df, manifest = load_feature_artifacts(load_input)
        if has_filters:
            cleaned_df = filter_feature_dataframe(cleaned_df, filters=filters)

        load_path = Path(load_input).resolve()
        source_dir = load_path.parent if load_path.is_file() else load_path
        print(f"Loaded artifacts from: {source_dir}")
        print(f"Feature dictionaries: {len(feature_dicts)}")
        print(f"Dataframe shape: {cleaned_df.shape}")
        if has_filters:
            print(f"Applied filters: {filters.__dict__}")
        if not feature_dicts:
            print("dictionary.json not found (continuing with dataframe-only mode).")
        if manifest:
            print("Manifest found.")
        if args.preview:
            _print_preview(cleaned_df)
        if cleaned_df.empty:
            raise SystemExit(
                "No rows matched the selected filters on loaded features."
            )
        return

    if not args.root:
        raise SystemExit("Please provide dataset root path or set STAT_AI_DATA.")

    paths = data_preprocessing(
        root=args.root,
        output_dir=args.out,
        limit=args.limit,
        only_active=not args.include_inactive,
        verbose=not args.quiet,
        filters=filters if has_filters else None,
    )

    print("Saved feature artifacts:")
    print(f"- output_dir: {paths['output_dir']}")
    print(f"- dictionary: {paths['dictionary']}")
    print(f"- cleaned_df: {paths['cleaned_df']}")
    print(f"- manifest: {paths['manifest']}")

    if args.preview:
        _, cleaned_df, _ = load_feature_artifacts(args.out)
        _print_preview(cleaned_df)


if __name__ == "__main__":
    main()
