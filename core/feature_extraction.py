import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis, entropy

def _to_float(val):
    """Safely convert numpy or array values to Python float."""
    if isinstance(val, (np.ndarray, list)):
        if len(np.shape(val)) == 0:
            return float(val)
        if len(val) == 1:
            return float(val[0])
        return float(np.mean(val))  # fallback for multi-element scalar signals
    try:
        return float(val)
    except Exception:
        return np.nan

def extract_vector_stats(signal: np.ndarray, prefix: str) -> dict:
    """Compute mean/std/rms/etc. over a 3D signal vector, flattening per-axis stats."""
    axes = ["x", "y", "z"]
    out = {}

    # Defensive: handle cases where signal doesn't have exactly 3 columns
    if signal.ndim != 2 or signal.shape[1] not in (2, 3):
        raise ValueError(f"Unexpected vector shape for {prefix}: {signal.shape}")

    # Compute stats
    mean_vals = np.mean(signal, axis=0)
    std_vals = np.std(signal, axis=0)
    rms_vals = np.sqrt(np.mean(signal**2, axis=0))
    ptp_vals = np.ptp(signal, axis=0)

    # Flatten each dimension explicitly
    for i, axis in enumerate(axes[: signal.shape[1]]):
        out[f"{prefix}_mean_{axis}"] = float(mean_vals[i])
        out[f"{prefix}_std_{axis}"] = float(std_vals[i])
        out[f"{prefix}_rms_{axis}"] = float(rms_vals[i])
        out[f"{prefix}_ptp_{axis}"] = float(ptp_vals[i])

    return out


def extract_scalar_stats(signal: np.ndarray, prefix: str) -> dict:
    """Compute time-domain features for 1D scalar signals."""
     # Avoid unstable skew/kurtosis on nearly constant data
    if np.allclose(signal, signal[0]):
        skew_val = np.nan
        kurt_val = np.nan
    else:
        try:
            skew_val = _to_float(skew(signal, bias=False, nan_policy="omit"))
        except Exception:
            skew_val = np.nan
        try:
            kurt_val = _to_float(kurtosis(signal, bias=False, nan_policy="omit"))
        except Exception:
            kurt_val = np.nan

    return {
        f"{prefix}_mean": _to_float(np.mean(signal)),
        f"{prefix}_std": _to_float(np.std(signal)),
        f"{prefix}_min": _to_float(np.min(signal)),
        f"{prefix}_max": _to_float(np.max(signal)),
        f"{prefix}_median": _to_float(np.median(signal)),
        f"{prefix}_rms": _to_float(np.sqrt(np.mean(signal ** 2))),
        f"{prefix}_skew": skew_val,
        f"{prefix}_kurt": kurt_val,
    }


def extract_features_from_bag(bag: dict) -> dict:
    sensor_type = bag["sensor_type"]
    df = bag["data"]
    odr = bag.get("odr", None)

    out = {
        "condition": bag["condition"],
        "belt_status": bag["belt_status"],
        "sensor": bag["sensor"],
        "sensor_type": sensor_type,
        "rpm": bag["rpm"],
    }

    # === VECTOR SIGNALS: ACC, GYRO, MAG ===
    # === VECTOR SIGNALS: ACC, GYRO, MAG ===
    if sensor_type in {"acc", "gyro", "mag"}:
        try:
            sig = df[["x", "y", "z"]].dropna().to_numpy()
            if sig.shape[0] < 50:
                print(f"⚠️ Too few samples in {bag['sensor']}")
                return out

            prefix = sensor_type
            out.update(extract_vector_stats(sig, prefix))

            if odr and odr > 100:
                fft_vals = np.abs(rfft(sig - sig.mean(axis=0), axis=0))
                freqs = rfftfreq(sig.shape[0], d=1.0 / odr)
                for i, axis in enumerate(["x", "y", "z"][:sig.shape[1]]):
                    out[f"{prefix}_dom_freq_{axis}"] = float(freqs[np.argmax(fft_vals[:, i])])
        except KeyError as e:
            print(f"⚠️ Missing expected axis in {bag['sensor']}: {e}")
            return out


    # === AUDIO (mic) ===
    elif sensor_type == "mic":
        # Detect all mic-like columns
        mic_cols = [c for c in df.columns if "mic" in c.lower() or "waveform" in c.lower() or "audio" in c.lower()]
        mic_cols = list(dict.fromkeys(mic_cols))  # Remove duplicates if any

        if not mic_cols:
            print(f"⚠️ No mic column found in {bag['sensor']}")
            return out

        # If there are multiple candidates, choose the one with highest standard deviation (i.e., most dynamic)
        if len(mic_cols) > 1:
            print(f"⚠️ Multiple mic-like columns found in {bag['sensor']}: {mic_cols}")
            try:
                stats = {col: df[col].std(skipna=True) for col in mic_cols}
                mic_col = max(stats, key=stats.get)
                print(f"✅ Using mic column with highest std: {mic_col}")
            except Exception as e:
                print(f"⚠️ Failed to pick mic column by std: {e}")
                mic_col = mic_cols[0]  # fallback
        else:
            mic_col = mic_cols[0]


        mic_series = df[mic_col].dropna()


        if mic_series.empty:
            return out
        #debug
        #mic_col_data = df[mic_col]
        # print(f"🧪 First few mic values from {bag['sensor']}:")
        # print(mic_col_data.head(5))
        # print("🔬 Column dtypes:\n", mic_col_data.dtypes)

        # If entries are array-like (lists, arrays), flatten them to a long 1D signal
        first_val = mic_series.iloc[0]
        if isinstance(first_val, (list, tuple, np.ndarray)):
            try:
                # Flatten all values into one long array
                sig = np.concatenate([np.asarray(x).ravel() for x in mic_series])
            except Exception as e:
                print(f"⚠️ mic flattening error: {e}")
                return out
        else:
            # Handle regular scalar series
            try:
                sig = pd.to_numeric(mic_series, errors="coerce").dropna().to_numpy()
            except Exception as e:
                print(f"⚠️ mic signal conversion error: {e}")
                return out

        if sig.size < 100:
            return out

        out.update(extract_scalar_stats(sig, "mic"))

        # Optional frequency domain features
        if odr and odr > 1000:
            fft_input = sig - np.mean(sig)
            fft_input *= np.hanning(len(fft_input))
            fft_vals = np.abs(rfft(fft_input))
            freqs = rfftfreq(len(fft_input), d=1.0 / odr)

            out["mic_dom_freq"] = _to_float(freqs[np.argmax(fft_vals)])
            psd = fft_vals ** 2
            p = psd / (np.sum(psd) + 1e-12)
            out["mic_freq_entropy"] = _to_float(entropy(p, base=2))




    # === SCALAR SIGNALS: TEMP, HUM, PRS ===
    elif sensor_type in {"temp", "hum", "prs"}:
        expected = {"temp": ["temp"], "hum": ["hum"], "prs": ["prs"]}.get(sensor_type, [])  
        for col in df.columns:
            if col in expected:
                sig = df[col].dropna().to_numpy()

                if sig.size < 2:
                    continue
                out.update(extract_scalar_stats(sig, col))

    return out

def extract_features_from_bags(bags: list[dict]) -> pd.DataFrame:
    rows = []
    for bag in bags:
        try:
            features = extract_features_from_bag(bag)
            rows.append(features)
        except Exception as e:
            print(f"⚠️ Error processing {bag.get('sensor')}: {e}")
    return pd.DataFrame(rows)