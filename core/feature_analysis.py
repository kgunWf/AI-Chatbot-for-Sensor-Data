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
from sklearn.model_selection import train_test_split
from core.feature_extraction import prepare_combined_feature_dataframe
#from catboost import CatBoostClassifier
#from xgboost import XGBClassifier

def analyze_global_features(df: pd.DataFrame, top_k=None, do_plots=True):
    """
    Global cross-sensor feature analysis.
    
    Modes:
    - Manual mode: user provides top_k (int) -> return that number of top features.
    - Auto mode: top_k=None -> automatically search the best k using validation accuracy.
    """

    # -----------------------------
    # 1) Build cleaned + imputed DF
    # -----------------------------

    meta_cols_requested = ["condition", "belt_status", "sensor", "sensor_type", "rpm"]
    meta_cols = [c for c in meta_cols_requested if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols]
    y = df["belt_status"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # -----------------------------
    # 2) Compute global importances
    # -----------------------------
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X, y_encoded)
    anova_scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)

    rfc = RandomForestClassifier(n_estimators=200, random_state=42)
    rfc.fit(X, y_encoded)
    rf_importances = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False)

    # -----------------------------
    # 3) Determine feature counts
    # -----------------------------
    all_k_values = [5, 10, 15, 20, 30, 40, len(feature_cols)]

    # If user provided a top_k -> manual mode
    manual_mode = top_k is not None

    if manual_mode:
        # Ensure top_k is <= total number of features
        top_k = min(top_k, len(rf_importances))
        best_k = top_k
    else:
        # Auto mode: search for best k
        accuracy_curve = {}

        for k in all_k_values:
            k = min(k, len(rf_importances))
            selected = rf_importances.head(k).index.tolist()

            Xk = X[selected]
            X_train, X_test, y_train, y_test = train_test_split(
                Xk, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
            )
            #model = XGBClassifier(
            #     n_estimators=400,
            #     learning_rate=0.05,
            #     max_depth=5,
            #     subsample=0.9,
            #     colsample_bytree=0.7,
            #     eval_metric="logloss",
            #     random_state=42
            # )

            #model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0)
            model = RandomForestClassifier(n_estimators=300, random_state=42)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            accuracy_curve[k] = acc

        # Best k = the one giving maximum accuracy
        best_k = max(accuracy_curve, key=accuracy_curve.get)

    # Now select features using best_k (manual OR auto)
    selected_features = rf_importances.head(best_k).index.tolist()

    # -----------------------------
    # 4) Plot if needed
    # -----------------------------
    if do_plots:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=rf_importances.head(best_k), y=rf_importances.head(best_k).index)
        plt.title(f"Top {best_k} Global Features (Random Forest)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

        if not manual_mode:  # Only plot accuracy curve in AUTO mode
            ks = list(accuracy_curve.keys())
            accs = list(accuracy_curve.values())
            plt.figure(figsize=(8, 5))
            plt.plot(ks, accs, marker="o")
            plt.title("Model Accuracy vs Number of Top Features")
            plt.xlabel("Number of Features")
            plt.ylabel("Test Accuracy")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # -----------------------------
    # 5) Return everything
    # -----------------------------
    out = {
        "anova_scores": anova_scores,
        "rf_importances": rf_importances,
        "top_features": selected_features,
        "selected_k": best_k,
        "mode": "manual" if manual_mode else "auto",
    }

    if not manual_mode:
        out["accuracy_curve"] = accuracy_curve
        out["best_accuracy"] = accuracy_curve[best_k]

    return out

if __name__ == "__main__":
    import pandas as pd
    print("Loading CSV...")
    
    try:
        df = pd.read_csv("results/features.csv")
    except FileNotFoundError:
        raise SystemExit("❌ Could not find results/features.csv. Extract features first.")

    print("Running global feature analysis...")
    res = analyze_global_features(df, top_k=None, do_plots=False)

    print("\n=== SUMMARY ===")
    print("Mode:", res["mode"])
    print("Selected k:", res["selected_k"])

    if "best_accuracy" in res:
        print("Best accuracy:", round(res["best_accuracy"], 3))

    print("\nTop features:")
    for f in res["top_features"]:
        print(" -", f)
        
    print("\nTop ANOVA F-scores:")
    print(res["anova_scores"].head(20))


