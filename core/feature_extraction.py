# features.py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import rfft, rfftfreq

TIME_COLS = {"timestamp", "time", "ts", "datetime", "label"}

def _numeric_signal_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns excluding time/label-like fields."""
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtypes, np.number)]
    return [c for c in num_cols if c.lower() not in TIME_COLS]

def extract_features_from_databag(bag: dict) -> dict:
    """
    Extract time & frequency-domain features from a single sensor databag.

    Expected bag schema (from our loaders):
      {
        'condition': str,
        'belt_status': str,
        'sensor': str,            # e.g. 'iis3dwb_acc'
        'rpm': str,               # e.g. 'PMI_100rpm'
        'data': pd.DataFrame,     # sensor samples
        'odr': float | None       # sampling rate in Hz (optional but needed for FFT features)
      }
    """
    features: dict = {
        "condition": bag.get("condition"),
        "belt_status": bag.get("belt_status"),
        "sensor": bag.get("sensor"),
        "rpm": bag.get("rpm"),
    }

    df: pd.DataFrame = bag.get("data")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return features  # nothing to do

    odr = bag.get("odr")  # <-- lowercase key from our loaders
    channels = _numeric_signal_columns(df)

    for axis in channels:
        # Safe numeric array
        sig = pd.to_numeric(df[axis], errors="coerce").astype(np.float64).values
        if sig.size == 0:
            continue
        # remove NaNs (if all NaN, skip)
        sig = sig[~np.isnan(sig)]
        if sig.size == 0:
            continue

        # -------- Time-domain --------
        features[f"{axis}_mean"]  = float(np.mean(sig))
        features[f"{axis}_std"]   = float(np.std(sig))
        features[f"{axis}_max"]   = float(np.max(sig))
        features[f"{axis}_min"]   = float(np.min(sig))
        features[f"{axis}_ptp"]   = float(np.ptp(sig))
        features[f"{axis}_rms"]   = float(np.sqrt(np.mean(sig ** 2)))
        # skew/kurtosis can fail on constant arrays — guard it:
        try:
            features[f"{axis}_skew"] = float(skew(sig, bias=False, nan_policy="omit"))
        except Exception:
            features[f"{axis}_skew"] = np.nan
        try:
            features[f"{axis}_kurt"] = float(kurtosis(sig, bias=False, nan_policy="omit"))
        except Exception:
            features[f"{axis}_kurt"] = np.nan

        # -------- Frequency-domain (only if odr is valid) --------
        if isinstance(odr, (int, float)) and odr > 0 and sig.size > 4:
            # zero-mean to reduce DC; simple Hann window to reduce leakage
            x = sig - np.mean(sig)
            w = np.hanning(x.size)
            xw = x * w

            fft_vals = np.abs(rfft(xw))
            freqs = rfftfreq(xw.size, d=1.0/float(odr))

            if fft_vals.size >= 2:
                # Dominant frequency (ignore the DC bin if present)
                start = 1 if freqs.size > 1 else 0
                dom_idx = np.argmax(fft_vals[start:]) + start
                dom_freq = float(freqs[dom_idx])
                features[f"{axis}_dom_freq"] = dom_freq

                # Spectral centroid
                denom = float(np.sum(fft_vals)) + 1e-12
                spectral_centroid = float(np.sum(freqs * fft_vals) / denom)
                features[f"{axis}_spec_centroid"] = spectral_centroid

                # Band energies (example split at 1 kHz; tweak as needed)
                low_mask = freqs < 1000.0
                high_mask = ~low_mask
                energy_low = float(np.sum((fft_vals[low_mask])**2))
                energy_high = float(np.sum((fft_vals[high_mask])**2))
                features[f"{axis}_band_energy_low"] = energy_low
                features[f"{axis}_band_energy_high"] = energy_high

                # Frequency entropy (Shannon on normalized PSD)
                psd = (fft_vals ** 2)
                psd_sum = float(np.sum(psd)) + 1e-12
                p = psd / psd_sum
                features[f"{axis}_freq_entropy"] = float(entropy(p, base=2))
        else:
            # Keep keys explicit if you prefer; or just skip adding freq features
            pass

    return features


def extract_features_from_bags(bags: list[dict]) -> pd.DataFrame:
    """Convenience: process many bags and return a tidy DataFrame."""
    rows = [extract_features_from_databag(b) for b in bags]
    return pd.DataFrame(rows)
