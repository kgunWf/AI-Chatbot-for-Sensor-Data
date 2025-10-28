import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import rfft, rfftfreq
from pandas.api.types import is_numeric_dtype

# Fields to ignore when extracting signals
TIME_COLS = {"timestamp", "time", "ts", "datetime", "label"}

def _looks_numeric_series(s: pd.Series) -> bool:
    s2 = (s.astype(str)
            .str.replace(",", ".", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0])
    s2 = pd.to_numeric(s2, errors="coerce")
    return s2.notna().sum() >= 2

def _numeric_signal_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        cl = c.lower()
        if cl in TIME_COLS:
            continue
        s = df[c]
        if is_numeric_dtype(s):
            if pd.to_numeric(s, errors="coerce").notna().sum() >= 2:
                cols.append(c)
        else:
            if _looks_numeric_series(s):
                cols.append(c)
    return cols

def _make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")  # temp, temp.1, temp.2, ...
        else:
            seen[c] = 0
            out.append(c)
    return out

def extract_features_from_databag(bag: dict) -> dict:
    """
    Extracts time-domain and frequency-domain features from a single databag.
    
    Output: one flat feature dict, ready to be used in a unified feature table.
    """
    # --- Basic metadata ---
    features = {
        "condition": bag.get("condition"),
        "belt_status": bag.get("belt_status"),
        "sensor": bag.get("sensor"),
        "sensor_type": bag.get("sensor_type", None),
        "rpm": bag.get("rpm"),
    }

    df = bag.get("data")
    odr = bag.get("odr")

    if not isinstance(df, pd.DataFrame) or df.empty:
        return features  # No signal to extract
    # NEW: ensure duplicate labels don’t break dtype logic
    df = df.copy()
    df.columns = _make_unique_columns(list(df.columns))
    #gets a list of column names that are numeric
    channels = _numeric_signal_columns(df)

    for axis in channels:
        ###burda bir sey olabilir
        raw = df[axis]
        if is_numeric_dtype(raw):
            s = pd.to_numeric(raw, errors="coerce")
        else:
            s = pd.to_numeric(
                raw.astype(str)
                .str.replace(",", ".", regex=False)
                .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0],
                errors="coerce"
            )
        sig = s.astype("float64").dropna().values
        if sig.size < 2:
            continue

        # --- Time-domain ---
        # --- Time-domain (always compute if valid size) ---
        features[f"{axis}_mean"] = np.mean(sig)
        features[f"{axis}_std"] = np.std(sig)
        features[f"{axis}_max"] = np.max(sig)
        features[f"{axis}_min"] = np.min(sig)
        features[f"{axis}_ptp"] = np.ptp(sig)
        features[f"{axis}_rms"] = np.sqrt(np.mean(sig ** 2))
        features[f"{axis}_median"] = np.median(sig)

        # Skew/kurtosis can fail for constant signals — guard it
        try:
            features[f"{axis}_skew"] = skew(sig, bias=False, nan_policy="omit")
        except Exception:
            features[f"{axis}_skew"] = np.nan

        try:
            features[f"{axis}_kurt"] = kurtosis(sig, bias=False, nan_policy="omit")
        except Exception:
            features[f"{axis}_kurt"] = np.nan


        # --- Frequency-domain ---
        if isinstance(odr, (int, float)) and odr > 0 and sig.size > 4:
            x = sig - np.mean(sig)
            w = np.hanning(x.size)
            xw = x * w

            fft_vals = np.abs(rfft(xw))
            freqs = rfftfreq(len(xw), d=1.0/odr)

            if fft_vals.size >= 2:
                dom_freq = freqs[np.argmax(fft_vals[1:]) + 1]  # Skip DC
                features[f"{axis}_dom_freq"] = dom_freq

                spec_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-12)
                features[f"{axis}_spec_centroid"] = spec_centroid

                # Band energy split at 1kHz
                low_mask = freqs < 1000.0
                features[f"{axis}_band_energy_low"] = np.sum(fft_vals[low_mask]**2)
                features[f"{axis}_band_energy_high"] = np.sum(fft_vals[~low_mask]**2)

                # Frequency entropy
                psd = fft_vals ** 2
                p = psd / (np.sum(psd) + 1e-12)
                features[f"{axis}_freq_entropy"] = entropy(p, base=2)

    return features

def extract_features_from_bags(bags: list[dict]) -> pd.DataFrame:
    """
    Extract features from a list of databags and return a unified dataframe.
    Missing/irrelevant features will appear as NaN.
    """
    rows = [extract_features_from_databag(b) for b in bags]
    return pd.DataFrame(rows)


