# core/odr_from_cfg.py
import json
from pathlib import Path

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
