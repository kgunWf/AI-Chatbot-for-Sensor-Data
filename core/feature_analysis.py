from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def analyze_features_from_bags(bags: list[dict], top_k=10, do_plots=True):
    # Group raw dicts by sensor_type
    grouped = defaultdict(list)
    for row in bags:
        if 'sensor_type' in row:
            grouped[row['sensor_type']].append(row)

    results = {}

    for sensor_type, entries in grouped.items():
        print(f"\n🔍 Analyzing sensor_type: {sensor_type} ({len(entries)} samples)")

        # Merge KO labels to binary
        for e in entries:
            e["belt_status"] = "KO" if str(e.get("belt_status", "")).upper().startswith("KO") else "OK"

        # Build DataFrame (only includes relevant features for this sensor_type)
        df = pd.DataFrame(entries)

        # Separate features and metadata
        meta_cols = ["condition", "belt_status", "sensor", "sensor_type", "rpm", "path"]
        feature_cols = [col for col in df.columns if col not in meta_cols]

        if not feature_cols:
            print(f"⚠️ No features found for sensor type {sensor_type}. Available columns: {df.columns.tolist()}")
            continue

        X_raw = df[feature_cols]
        y = df["belt_status"]

        if len(y.unique()) < 2:
            print("⚠️ Skipping - only one class present")
            continue

        # Verify we have actual data to process
        if X_raw.empty or X_raw.isnull().all().all():
            print(f"⚠️ No valid feature data for sensor type {sensor_type}")
            continue

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Defensive numeric conversion + cleaning before imputation
        # Coerce to numeric (turn strings -> NaN), replace infs with NaN, clip extreme values
        X_num = X_raw.copy()
        # attempt to convert all columns to numeric where possible
        X_num = X_num.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        # count problematic values
        n_infs = int(np.isinf(X_num.values).sum())
        n_nonfinite = int(~np.isfinite(X_num.values).sum())
        if n_infs or n_nonfinite:
            print(f"⚠️ Found non-finite values in features for {sensor_type}: infs={n_infs}, non-finite={n_nonfinite}")
        # replace +/-inf with NaN
        X_num.replace([np.inf, -np.inf], np.nan, inplace=True)
        # optionally cap extreme magnitudes to avoid overflow in downstream (use safe threshold)
        VERY_LARGE = 1e200
        mask_large = X_num.abs() > VERY_LARGE
        if mask_large.values.any():
            print(f"⚠️ Found values exceeding {VERY_LARGE} in {sensor_type}; setting them to NaN to avoid dtype overflow")
            X_num = X_num.mask(mask_large)

        # drop columns that are entirely NaN after coercion
        cols_before = X_num.shape[1]
        X_num.dropna(axis=1, how="all", inplace=True)
        cols_after = X_num.shape[1]
        if cols_after < cols_before:
            print(f"ℹ️ Dropped {cols_before-cols_after} all-NaN feature columns for {sensor_type}")

        if X_num.empty or X_num.isnull().all().all():
            print(f"⚠️ No numeric feature data left for sensor type {sensor_type} after cleaning")
            continue

        # Impute missing values (should be few or none now)
        X = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X_num), columns=X_num.columns)

        # Feature importance
        k_best = SelectKBest(score_func=f_classif, k="all")
        k_best.fit(X, y_encoded)
        k_scores = pd.Series(k_best.scores_, index=X.columns).sort_values(ascending=False)

        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X, y_encoded)
        y_pred = rfc.predict(X)
        acc = accuracy_score(y_encoded, y_pred)
        rf_importances = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False)

        if do_plots:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=rf_importances.head(top_k), y=rf_importances.head(top_k).index)
            plt.title(f"Top {top_k} Features - {sensor_type} (RF Importance)")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.show()

            # PCA
            try:
                X_pca = PCA(n_components=2).fit_transform(X)
                plt.figure(figsize=(7, 5))
                sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y)
                plt.title(f"PCA - {sensor_type}")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"⚠️ PCA failed for {sensor_type}: {e}")

        results[sensor_type] = {
            "accuracy": acc,
            "rf_importances": rf_importances,
            "anova_scores": k_scores,
            "top_features": rf_importances.head(top_k).index.tolist(),
            "X": X,
            "y": y_encoded,
            "df": df
        }

        print(f"✅ {sensor_type} accuracy: {acc:.2f}")

    return results
