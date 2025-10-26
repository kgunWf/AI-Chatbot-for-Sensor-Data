# core/odr_from_cfg.py
import json
from pathlib import Path
import pandas as pd
import re

import re


def normalize_sensor_columns(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """
    Standardize column names like 'A_x [g]', 'TEMP [C]' to 'x', 'temp', etc.
    Keeps multiple axes if present.
    """
    new_cols = {}
    for col in df.columns:
        col_lower = col.strip().lower()

        # Map common types
        if "x" in col_lower and "mag" in col_lower or "x" in col_lower and "acc" in col_lower or "x" == col_lower:
            new_cols[col] = "x"
        elif "y" in col_lower and "mag" in col_lower or "y" in col_lower and "acc" in col_lower or "y" == col_lower:
            new_cols[col] = "y"
        elif "z" in col_lower and "mag" in col_lower or "z" in col_lower and "acc" in col_lower or "z" == col_lower:
            new_cols[col] = "z"
        elif "temp" in col_lower:
            new_cols[col] = "temp"
        elif "hum" in col_lower:
            new_cols[col] = "hum"
        elif "press" in col_lower or "prs" in col_lower:
            new_cols[col] = "prs"
        elif "mic" in col_lower or "aud" in col_lower:
            new_cols[col] = "mic"
        else:
            new_cols[col] = col_lower  # fallback

    return df.rename(columns=new_cols)


# normalize names like "IIS3DWB_ACC" → "iis3dwb_acc"
def norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

# map verbose subtype names to your suffix style
SUB = {
    "accelerometer":"acc","accel":"acc","acc":"acc",
    "gyroscope":"gyro","gyro":"gyro",
    "magnetometer":"mag","mag":"mag",
    "temperature":"temp","temp":"temp",
    "humidity":"hum","hum":"hum",
    "pressure":"press","press":"press",
    "microphone":"mic","audio":"mic","mic":"mic",
}

def get_sensor_type(sensor_name: str) -> str:
    """
    Infers the sensor type (acc, gyro, mic, etc.) from a full sensor or sensor+subtype name.
    Examples:
        "IIS3DWB_ACC" → "acc"
        "HTS221_TEMP" → "temp"
    """
    parts = norm(sensor_name).split("_")
    for p in reversed(parts):
        if p in SUB:
            return SUB[p]
    return "unknown"

def get_odr_map(acq_dir: str | Path) -> dict[str, float]:
    """Return {'iis3dwb_acc': 26667.0, 'hts221_temp': 12.5, ...} from STWIN_*/DeviceConfig.json."""
    cfg = Path(acq_dir) / "DeviceConfig.json"
    if not cfg.exists():
        return {}
    j = json.loads(cfg.read_text(encoding="utf-8"))
    sensors = (j.get("device") or {}).get("sensor") or []
    out = {}
    for s in sensors:
        name = norm(s.get("name",""))
        desc = (s.get("sensorDescriptor") or {}).get("subSensorDescriptor") or []
        stat = (s.get("sensorStatus")     or {}).get("subSensorStatus")     or []
        for d, st in zip(desc, stat):
            subtype = SUB.get(norm(d.get("sensorType","")), None)
            key = f"{name}_{subtype}" if subtype else name
            # prefer measured ODR, else configured ODR; keep it simple
            odr = st.get("ODRMeasured") or st.get("ODR")
            if odr is not None:
                try: out[key] = float(odr)
                except: pass
    return out
