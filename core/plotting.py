# core/plot_raw.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from data_loader import load_raw_bags, filter_bags,group_by_sensor_name

def plot_frequency_spectrum(bag: dict, axis: str | None = None):
    """
    Unified, robust FFT plot for ANY supported sensor bag.

    - acc/gyro/mag → FFT of x/y/z (axis="x"/"y"/"z"; axis=None => all)
    - mic          → FFT of flattened waveform
    - temp/hum/prs → skipped (not meaningful)

    Uses ODR as the sampling rate.
    """

    sensor_type = bag.get("sensor_type")
    sensor_name = bag.get("sensor")
    df: pd.DataFrame = bag.get("data")
    odr = bag.get("odr", None)

    # You NEED sampling frequency for FFT
    if not odr or odr <= 0:
        print(f"⚠️ Skipping {sensor_name}: no valid ODR (needed for FFT).")
        return

    # -------------------------
    # VECTOR SENSORS (acc/gyro/mag)
    # -------------------------
    if sensor_type in {"acc", "gyro", "mag"}:
        axes = ["x", "y", "z"]
        if not all(a in df.columns for a in axes):
            print(f"⚠️ Skipping {sensor_name}: missing x/y/z columns.")
            return

        vec = df[axes].dropna()
        if vec.empty:
            print(f"⚠️ Skipping {sensor_name}: no x/y/z samples.")
            return

        n = len(vec)

        # Time spacing = 1/odr
        freqs = rfftfreq(n, d=1.0 / odr)

        plt.figure()

        if axis in axes:  # Single axis FFT
            sig = vec[axis].to_numpy()
            fft_vals = np.abs(rfft(sig - sig.mean()))
            plt.plot(freqs, fft_vals, label=f"{sensor_type}_{axis}")

        else:  # All axes FFT
            for ax in axes:
                sig = vec[ax].to_numpy()
                fft_vals = np.abs(rfft(sig - sig.mean()))
                plt.plot(freqs, fft_vals, label=f"{sensor_type}_{ax}")

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(f"Frequency Spectrum – {sensor_name} [{sensor_type}]")
        plt.xlim(0, odr/2)   # show up to Nyquist
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # -------------------------
    # MIC SENSORS
    # -------------------------
    if sensor_type == "mic":
        # find mic-like columns
        mic_cols = [
            c for c in df.columns
            if "mic" in c.lower() or "audio" in c.lower() or "waveform" in c.lower()
        ]
        mic_cols = list(dict.fromkeys(mic_cols))

        if not mic_cols:
            print(f"⚠️ Skipping {sensor_name}: no mic-like columns.")
            return

        mic_col = mic_cols[0]
        series = df[mic_col].dropna()
        if series.empty:
            print(f"⚠️ Skipping {sensor_name}: mic data empty.")
            return

        # Flatten if necessary
        first = series.iloc[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            sig = np.concatenate([np.asarray(x).ravel() for x in series])
        else:
            sig = pd.to_numeric(series, errors="coerce").dropna().to_numpy()

        if sig.size == 0:
            print(f"⚠️ Skipping {sensor_name}: mic signal invalid.")
            return

        n = sig.size
        freqs = rfftfreq(n, d=1.0 / odr)
        fft_vals = np.abs(rfft(sig - np.mean(sig)))

        plt.figure()
        plt.plot(freqs, fft_vals)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(f"Frequency Spectrum – {sensor_name} [mic]")
        plt.xlim(0, odr/2)
        plt.tight_layout()
        plt.show()
        return

    # -------------------------
    # TEMP/HUM/PRS (skip)
    # -------------------------
    print(f"⚠️ Skipping {sensor_name}: FFT not applicable to sensor_type {sensor_type}.")



def _build_time_axis(n: int, odr: float | None):
    """Return (t, label) for the time axis."""
    if odr and odr > 0:#if odr(sampling rate) is available and greater than 0, use it to build the time axis
        t = np.arange(n) / odr
        label = "Time (s)"
    else:
        t = np.arange(n)#if odr(sampling rate) is not available or less than 0, use sample index to build the time axis
        label = "Sample index"
    return t, label

def is_bag_plottable(bag: dict) -> bool:
    """
    Check if this bag has the columns needed to be plotted,
    based on its sensor_type.
    """
    sensor_type = bag["sensor_type"]
    df = bag["data"]
    cols = list(df.columns)

    # Vector sensors: need x, y, z
    if sensor_type in {"acc", "gyro", "mag"}:
        return all(c in df.columns for c in ["x", "y", "z"])

    # Scalar sensors
    if sensor_type == "temp":
        return "temp" in df.columns
    if sensor_type == "hum":
        return "hum" in df.columns
    if sensor_type == "prs":
        return "prs" in df.columns

    # Mic: need at least one mic-like column
    if sensor_type == "mic":
        mic_cols = [
            c for c in df.columns
            if "mic" in c.lower() or "audio" in c.lower() or "waveform" in c.lower()
        ]
        return len(mic_cols) > 0

    # Anything else: not supported (for now)
    return False

def plot_time_series(bag: dict, axis: str | None = None) -> None:
    """
    Unified, robust time-series plot for ANY supported sensor bag.

    - acc/gyro/mag → x/y/z axes (axis="x"/"y"/"z"; axis=None => all 3)
    - temp/hum/prs → scalar column ("temp", "hum", "prs")
    - mic          → flattened waveform from mic/audio/waveform column

    If the bag is missing expected columns, the function prints a warning
    and returns without raising, so callers can safely loop.
    """
    sensor_type = bag.get("sensor_type")
    sensor_name = bag.get("sensor")
    df: pd.DataFrame = bag.get("data")
    odr = bag.get("odr", None)

    if df is None:
        print(f"⚠️ Skipping {sensor_name}: no 'data' DataFrame in bag.")
        return

    # -------------------------
    # VECTOR SENSORS: acc/gyro/mag
    # -------------------------
    if sensor_type in {"acc", "gyro", "mag"}:
        axes = ["x", "y", "z"]
        if not all(a in df.columns for a in axes):
            print(f"⚠️ Skipping {sensor_name}: missing x,y,z columns; cols={list(df.columns)}")
            return

        vec = df[axes].dropna()
        if vec.empty:
            print(f"⚠️ Skipping {sensor_name}: no x/y/z samples after dropna().")
            return

        n = len(vec)
        t, t_label = _build_time_axis(n, odr)

        plt.figure()

        if axis in axes:  # plot a single axis
            arr = vec[axis].to_numpy()
            plt.plot(t, arr, label=f"{sensor_type}_{axis}")
        else:             # plot all three axes
            for ax in axes:
                plt.plot(t, vec[ax].to_numpy(), label=f"{sensor_type}_{ax}")

        plt.xlabel(t_label)
        plt.ylabel(sensor_type)
        plt.title(f"Time series – {sensor_name} [{sensor_type}]")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # -------------------------
    # SCALAR SENSORS: temp/hum/prs
    # -------------------------
    if sensor_type in {"temp", "hum", "prs"}:
        col_map = {"temp": "temp", "hum": "hum", "prs": "prs"}
        col = col_map[sensor_type]

        if col not in df.columns:
            print(f"⚠️ Skipping {sensor_name}: expected '{col}' not in {list(df.columns)}")
            return

        sig = df[col].dropna().to_numpy()
        if sig.size == 0:
            print(f"⚠️ Skipping {sensor_name}: column '{col}' is empty after dropna().")
            return

        n = sig.size
        t, t_label = _build_time_axis(n, odr)

        plt.figure()
        plt.plot(t, sig)
        plt.xlabel(t_label)
        plt.ylabel(col)
        plt.title(f"Time series – {sensor_name} [{sensor_type}]")
        plt.tight_layout()
        plt.show()
        return

    # -------------------------
    # MIC SENSORS
    # -------------------------
    if sensor_type == "mic":
        mic_cols = [
            c for c in df.columns
            if "mic" in c.lower() or "audio" in c.lower() or "waveform" in c.lower()
        ]
        mic_cols = list(dict.fromkeys(mic_cols))  # dedup, preserve order

        if not mic_cols:
            print(f"⚠️ Skipping {sensor_name}: no mic-like columns; cols={list(df.columns)}")
            return

        # If multiple candidates, pick the one with highest std
        if len(mic_cols) > 1:
            try:
                stats = {c: df[c].std(skipna=True) for c in mic_cols}
                mic_col = max(stats, key=stats.get)
            except Exception as e:
                print(f"⚠️ Skipping {sensor_name}: cannot choose mic column by std ({e}); cols={mic_cols}")
                return
        else:
            mic_col = mic_cols[0]

        series = df[mic_col].dropna()
        if series.empty:
            print(f"⚠️ Skipping {sensor_name}: mic column '{mic_col}' is empty after dropna().")
            return

        first = series.iloc[0]
        try:
            if isinstance(first, (list, tuple, np.ndarray)):
                # Each row is a small vector → flatten all of them
                sig = np.concatenate([np.asarray(x).ravel() for x in series])
            else:
                # Standard scalar 1D signal
                sig = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        except Exception as e:
            print(f"⚠️ Skipping {sensor_name}: error while flattening mic data ({e}).")
            return

        if sig.size == 0:
            print(f"⚠️ Skipping {sensor_name}: mic signal is empty after conversion.")
            return

        n = sig.size
        t, t_label = _build_time_axis(n, odr)

        plt.figure()
        plt.plot(t, sig)
        plt.xlabel(t_label)
        plt.ylabel(mic_col)
        plt.title(f"Time series – {sensor_name} [mic]")
        plt.tight_layout()
        plt.show()
        return

    # -------------------------
    # UNSUPPORTED SENSOR TYPE
    # -------------------------
    print(f"⚠️ Skipping {sensor_name}: unsupported sensor_type '{sensor_type}'.")

#function that plots the bags according to the provided filters
#we assume that the bags are loaded somewhere else
def time_plot(
    bags: list[dict],
    sensor_type: str | None = None,
    sensor: str | None = None,
    belt_status: str | None = None,
    condition: str | None = None,
    rpm: str |None=None,
) -> list[dict]:
    #root = os.getenv("STAT_AI_DATA", "/Users/zeynepoztunc/Downloads/Sensor_STWIN")
    #bags = load_raw_bags(root, verbose=False)
    #print("Total bags loaded:", len(bags))

    # Example: KO acc sensors → plot all axes
    filtered_bags = filter_bags(bags, sensor_type, belt_status,rpm,condition)

    #group filtered sensor by their names (such as iis3dwb_acc', 'iis2dh_acc', 'ism330dhcx_acc' )
    grouped_acc = group_by_sensor_name(filtered_bags)#this is a dictionary

    representatives = {
         name: cycles[0] #key = sensor name and value = the first recording for that sensor
         for name, cycles in grouped_acc.items()
     }

    for bag in representatives.values():
         plot_time_series(bag)

def freq_plot(
    bags: list[dict],
    sensor_type: str | None = None,
    sensor: str | None = None,
    belt_status: str | None = None,
    condition: str | None = None,
    rpm: str |None=None,
) -> list[dict]:
    #root = os.getenv("STAT_AI_DATA", "/Users/zeynepoztunc/Downloads/Sensor_STWIN")
    #bags = load_raw_bags(root, verbose=False)
    #print("Total bags loaded:", len(bags))

    # Example: KO acc sensors → plot all axes
    filtered_bags = filter_bags(bags, sensor_type="temp", belt_status="KO_LOW_4mm",rpm="PMI_50rpm",condition="vel-fissa")

    #group filtered sensor by their names (such as iis3dwb_acc', 'iis2dh_acc', 'ism330dhcx_acc' )
    grouped_acc = group_by_sensor_name(filtered_bags)#this is a dictionary

    representatives = {
         name: cycles[0] #key = sensor name and value = the first recording for that sensor
         for name, cycles in grouped_acc.items()
     }

    for bag in representatives.values():
         plot_time_series(bag)