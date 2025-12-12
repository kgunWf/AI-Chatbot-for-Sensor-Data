"""
Standalone script to test tools directly WITHOUT the agent.
This validates:
- cleaned_df loading
- build_tools()
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
from core.plotting import plot_time_series, plot_frequency_spectrum
from core.feature_analysis import analyze_global_features
from agent.tools import build_tools

# -------------------------------------------------
# Streamlit mock (important)
# -------------------------------------------------
class MockStreamlit:
    @staticmethod
    def pyplot(fig):
        print(f"[MockStreamlit] pyplot called with {type(fig)}")

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


# =================================================
# Tests
# =================================================

def load_cleaned_df():
    logger.info("Loading cleaned dataframe...")
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded DF with shape {df.shape}")
    return df


def test_core_feature_analysis(df):
    print("\n" + "=" * 60)
    print("TEST: analyze_global_features(df)")
    print("=" * 60)

    start = datetime.now()
    result = analyze_global_features(df=df, do_plots=True)
    elapsed = (datetime.now() - start).total_seconds()

    assert result is not None
    assert result.get("top_features") is not None

    print(f"✅ Completed in {elapsed:.2f}s")
    print(result["top_features"][:10])


def test_core_plotting(df):
    print("\n" + "=" * 60)
    print("TEST: plot_time_series(df, ...)")
    print("=" * 60)

    start = datetime.now()
    fig = plot_time_series(
        df=df,
        sensor="acc",
        condition="vel-fissa",
        belt_status="OK",
        stwin=None,
    )
    elapsed = (datetime.now() - start).total_seconds()

    assert fig is not None
    print(f"✅ Figure created in {elapsed:.2f}s")


def test_tools_via_build_tools(df):
    print("\n" + "=" * 60)
    print("TEST: build_tools(df) + tool invocation")
    print("=" * 60)

    tools = build_tools(df)
    tool_map = {tool.name: tool.func for tool in tools}

    # ---- InspectDataset
    print("\n▶ InspectDataset")
    result = tool_map["InspectDataset"]("")
    print(result)

    # ---- FeatureImportance
    print("\n▶ FeatureImportance")
    result = tool_map["FeatureImportance"]("")
    print(result)

    # ---- PlotSensor
    print("\n▶ PlotSensor")
    result = tool_map["PlotSensor"]("acc OK time vel-fissa")
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
    test_core_plotting(df)
    test_tools_via_build_tools(df)

    print("\n" + "=" * 60)
    print("🎉 ALL TOOL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
