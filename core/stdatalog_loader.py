
# stdatalog_hsd_loader.py
# -----------------------------------------------------------------------------
# Loader that calls the same STDatalog Core APIs used in your notebook:
#   from stdatalog_core.HSD.HSDatalog import HSDatalog
#   hsd.validate_hsd_folder, hsd.create_hsd, hsd.get_sensor_list, hsd.get_sensor,
#   hsd.get_dataframe / get_dataframe_gen
#
# Yields items shaped as:
# {
#   'condition': 'vel-fissa',
#   'belt_status': 'OK',
#   'sensor': 'IIS3DWB',
#   'rpm': 'PMI_100rpm',
#   'data': pd.DataFrame(...)
# }
#
# Usage:
#   from core.stdatalog_hsd_loader import iter_hsd_items
#   for item in iter_hsd_items('/path/to/dataset_root', verbose=True):
#       ...
#
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Iterable
from utils import get_odr_map, norm
import pandas as pd

# --- Optional: make local SDK repo importable if it's inside the project tree ---
# (This mirrors what your notebook does by pushing '../../' on sys.path.)
import sys as _sys
_here = Path(__file__).resolve()
for rel in ['..', '../..']:
    cand = (_here.parent / rel).resolve()
    if cand.exists() and str(cand) not in _sys.path:
        _sys.path.append(str(cand))
# -------------------------------------------------------------------------------

try:
    from stdatalog_core.HSD.HSDatalog import HSDatalog  # SDK class used in your notebook
except Exception as _e:
    HSDatalog = None
    _IMPORT_ERR = _e
else:
    _IMPORT_ERR = None

# ---------------------- path metadata inference helpers ------------------------
KNOWN_CONDITIONS = {'vel-fissa', 'no-load-cycles', 'vel_fissa', 'no_load_cycles'}
RPM_RE = re.compile(r'^(PMI|PMS)[_-]?(\d+\s*rpm|\d+rpm|\d+)$', re.IGNORECASE)

# --- add near top of stdatalog_hsd_loader.py ---


def _find_token(parts: Iterable[str], pool: set[str]) -> Optional[str]:
    for p in parts:
        if p in pool:
            return p
    return None

def _find_status(parts: Iterable[str]) -> Optional[str]:
    for p in parts:
        up = p.upper()
        if up == 'OK' or up.startswith('KO'):
            return p
    return None

def _find_rpm(parts: Iterable[str]) -> Optional[str]:
    norm = [p.replace('-', '_') for p in parts]
    for p in norm:
        m = RPM_RE.match(p)
        if m:
            kind = m.group(1).upper()
            num = re.sub(r'[^0-9]', '', m.group(2))
            return f"{kind}_{num}rpm"
    for i in range(len(norm)-1):
        a, b = norm[i], norm[i+1]
        if a.upper() in {'PMI','PMS'} and re.search(r'\d+', b):
            num = re.sub(r'[^0-9]', '', b)
            return f"{a.upper()}_{num}rpm"
    return None

def _infer_meta_from_path(acq_dir: Path) -> Dict[str, str]:
    parts = acq_dir.parts
    condition = _find_token(parts, KNOWN_CONDITIONS) or 'vel-fissa'
    status = _find_status(parts) or 'OK'
    rpm = _find_rpm(parts) or 'PMI_100rpm'
    if condition == 'vel_fissa':
        condition = 'vel-fissa'
    return {'condition': condition, 'belt_status': status, 'rpm': rpm}
# -------------------------------------------------------------------------------

def _is_hsd_folder(dir_path: Path) -> bool:
    # Heuristics: presence of device config json or binary sensor files
    if not dir_path.is_dir():
        return False
    for name in ['deviceConfig.json', 'DeviceConfig.json', 'hsd.json']:
        if (dir_path / name).exists():
            return True
    # Fallback: contains at least one .dat file
    return any(p.suffix.lower()=='.dat' for p in dir_path.iterdir() if p.is_file())

def _normalize_dataframe(obj) -> Optional[pd.DataFrame]:
    """Accept DataFrame or dict/list of DataFrames and return a single DataFrame."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        parts = [df for df in obj.values() if isinstance(df, pd.DataFrame)]
        if not parts:
            return None
        return pd.concat(parts, axis=1).sort_index()
    if isinstance(obj, (list, tuple)):
        parts = [df for df in obj if isinstance(df, pd.DataFrame)]
        if not parts:
            return None
        return pd.concat(parts, axis=1).sort_index()
    return None

def _iter_acquisition_dirs(root: Path) -> Iterator[Path]:
    # Consider leaf directories named STWIN_* or any dir that passes _is_hsd_folder
    for d in root.rglob('*'):
        if d.is_dir() and (d.name.upper().startswith('STWIN_') or _is_hsd_folder(d)):
            if _is_hsd_folder(d):
                yield d

def iter_hsd_items(root: str | Path, only_active: bool=True, verbose: bool=False) -> Iterator[Dict[str, object]]:
    if HSDatalog is None:
        raise ImportError(f"Could not import HSDatalog from stdatalog_core: {_IMPORT_ERR}")
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    for acq_dir in _iter_acquisition_dirs(root):
        meta = _infer_meta_from_path(acq_dir)
        try:
            hsd = HSDatalog()
            try:
                _ = hsd.validate_hsd_folder(str(acq_dir))
            except Exception as e:
                if verbose:
                    print(f"⚠️  Invalid HSD folder {acq_dir}: {e}")
                continue
            hsd_instance = hsd.create_hsd(acquisition_folder=str(acq_dir))

            # Build ODR map once per folder
            odr_map = get_odr_map(acq_dir)

            # list sensors
            sensor_names = hsd.get_sensor_list(hsd_instance, only_active=only_active)
            if sensor_names and not isinstance(sensor_names[0], str):
                sensor_names = [hsd.get_sensor_name(hsd_instance, s) for s in sensor_names]

            for sensor_name in sensor_names:
                sensor = hsd.get_sensor(hsd_instance, sensor_name)
                df_obj = hsd.get_dataframe(hsd_instance, sensor)
                df = _normalize_dataframe(df_obj)
                if df is None:
                    continue

                norm_sensor = norm(str(sensor_name))
                odr = odr_map.get(norm_sensor)

                yield {
                    'condition': meta['condition'],
                    'belt_status': meta['belt_status'],
                    'sensor': str(sensor_name),
                    'rpm': meta['rpm'],
                    'data': df,
                    'odr': odr,  # ✅ added ODR field
                }

        except Exception as e:
            if verbose:
                print(f"⚠️  Skipping acquisition {acq_dir}: {e}")
            continue


__all__ = ['iter_hsd_items']