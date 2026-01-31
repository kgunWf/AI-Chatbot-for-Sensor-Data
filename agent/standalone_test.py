"""
Standalone script to test tools directly WITHOUT the agent.

Validates:
- cleaned_df loading
- raw plotting via plot_raw_wrapper
- build_tools(df, raw_root)
- plot + feature tools execution
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# -------------------------------------------------
# Path setup
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "core"
AGENT_DIR = PROJECT_ROOT / "agent"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(AGENT_DIR))

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tool-tests")

# -------------------------------------------------
# Imports AFTER path setup
# -------------------------------------------------
from core.plotting import plotting
from core.feature_analysis import analyze_global_features
from agent.tools import build_tools

# -------------------------------------------------
# Streamlit mock
# -------------------------------------------------
class MockStreamlit:
    @staticmethod
    def pyplot(fig=None):
        print("[MockStreamlit] pyplot called")

    @staticmethod
    def write(text):
        print(f"[MockStreamlit] write: {text}")

    @staticmethod
    def table(data):
        shape = data.shape if hasattr(data, "shape") else "unknown"
        print(f"[MockStreamlit] table with shape {shape}")

sys.modules["streamlit"] = MockStreamlit()

# -------------------------------------------------
# Paths
# -------------------------------------------------
CSV_PATH = PROJECT_ROOT / "core" / "output_dir" / "cleaned_df.csv"

# 🔴 IMPORTANT: raw data root
RAW_DATA_ROOT = Path("/Users/kgun/Downloads/Sensor_STWIN")
# Example:
# /Users/.../Sensor_STWIN


# =================================================
# Helpers
# =================================================

def load_cleaned_df():
    logger.info("Loading cleaned dataframe...")
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded DF with shape {df.shape}")
    return df


# =================================================
# Tests
# =================================================

def test_core_feature_analysis(df):
    print("\n" + "=" * 60)
    print("TEST: analyze_global_features(df)")
    print("=" * 60)

    start = datetime.now()
    result = analyze_global_features(df=df, do_plots=False)
    elapsed = (datetime.now() - start).total_seconds()

    assert result is not None
    assert result.get("top_features") is not None

    print(f"✅ Completed in {elapsed:.2f}s")
    print(result["top_features"][:10])


def test_core_plotting_raw(raw_root: Path):
    print("\n" + "=" * 60)
    print("TEST: plot_raw_wrapper (raw data)")
    print("=" * 60)

    start = datetime.now()

    plotting(
        base_root=str(raw_root),
        plot_type="time",
        sensor_type="acc",
        belt_status="OK",
        condition="vel-fissa",
        rpm="PMS_50rpm",
    )

    elapsed = (datetime.now() - start).total_seconds()
    print(f"✅ Raw plotting completed in {elapsed:.2f}s")


def test_tools_via_build_tools(df, raw_root: Path):
    print("\n" + "=" * 60)
    print("TEST: build_tools(df, raw_root) + tool invocation")
    print("=" * 60)

    tools = build_tools(
        cleaned_df=df,
        path_to_data=str(raw_root)
    )

    tool_map = {tool.name: tool.func for tool in tools}

    # ---- InspectDataset
    print("\n▶ InspectDataset")
    result = tool_map["InspectDataset"]("")
    print(result)

    # ---- FeatureImportance
    print("\n▶ FeatureImportance")
    result = tool_map["FeatureImportance"]("")
    print(result)

    # ---- PlotSensor (RAW)
    print("\n▶ PlotSensor")
    result = tool_map["PlotSensor"]("acc OK time vel-fissa PMS_50rpm")
    print(result)


# =================================================
# Main
# =================================================

def main():
    print("\n" + "=" * 60)
    print("STANDALONE TOOL TEST SUITE")
    print("=" * 60)

    df = load_cleaned_df()

    test_core_feature_analysis(df)
    test_core_plotting_raw(RAW_DATA_ROOT)
    test_tools_via_build_tools(df, RAW_DATA_ROOT)

    print("\n" + "=" * 60)
    print("🎉 ALL TOOL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
