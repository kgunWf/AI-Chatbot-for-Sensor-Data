# core/data_loader.py
import os
import json
from typing import Iterable, Dict

from stdatalog_loader import iter_hsd_items


def load_hsd_items(root: str):
    # root can be your top-level dataset, e.g. "/Users/kgun/Downloads/Sensor_STWIN"
    for item in iter_hsd_items(root, only_active=True, verbose=True):
        yield item

def test_hsd_loader(data_root: str, limit: int = 10, verbose: bool = True) -> None:
    """
    Quick CLI-style test: scans `data_root`, uses the pysdk to read each STWIN acquisition,
    and prints a compact line per sensor dataframe.
    """
    n = 0
    for it in iter_hsd_items(data_root, only_active=True, verbose=verbose):
        n += 1
        print(f"{n:04d} | {it['condition']} | {it['belt_status']} | "
      f"{it['sensor']} | {it['rpm']} | rows={len(it['data'])} | odr={it.get('odr')}")

        #df = item["data"]
        #print(item["sensor"], df.shape, df.head(3))

        if n >= limit:
            break
    print(f"\nDone. Printed {min(n, limit)} of {n} loaded items.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Test HSD loader via core/data_loader.py")
    ap.add_argument("root", nargs="?", default=os.getenv("STAT_AI_DATA", ""), help="Dataset root path")
    ap.add_argument("--limit", type=int, default=10, help="Max items to print")
    ap.add_argument("--quiet", action="store_true", help="Less verbose")
    args = ap.parse_args()

    if not args.root:
        raise SystemExit("Provide a dataset root path or set STAT_AI_DATA")

    test_hsd_loader(args.root, limit=args.limit, verbose=(not args.quiet))
