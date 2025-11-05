import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def debug_temperature(bags):
    for i, b in enumerate(bags):
        df = b.get("data")
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"[bag {i}] empty dataframe")
            continue

        cols = list(df.columns)
        temp_cols = [c for c in cols if "temp" in c.lower() or "°" in c]
        if not temp_cols:
            continue  # not a temperature bag

        print(f"\n[bag {i}] sensor={b.get('sensor')} sensor_type={b.get('sensor_type')}")
        print("columns:", cols)
        print("dtypes:\n", df.dtypes)

        # What your current helper returns
        channels_current = [c for c in cols if np.issubdtype(df[c].dtypes, np.number) and c.lower() not in {"timestamp","time","ts","datetime","label"}]
        print("channels_current:", channels_current)

        # Try coercing any temp-like columns and count usable samples
        for c in temp_cols:
            raw = df[c]
            # If it’s already numeric, use it as is; otherwise coerce
            if is_numeric_dtype(raw):
                s = pd.to_numeric(raw, errors="coerce")
            else:
                s = pd.to_numeric(raw.astype(str).str.replace(",", ".", regex=False)
                                  .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0],
                                  errors="coerce")
            count = s.notna().sum()
            print(f"  {c}: usable_samples={count}, first_values={s.dropna().head(5).tolist()}")
