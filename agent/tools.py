# agent/tools.py
from langchain_core.tools import Tool
import logging
from datetime import datetime
import streamlit as st
import os
import sys
# Get the parent directory 
core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
if core_dir not in sys.path: 
    sys.path.insert(0, core_dir)
from core.plotting import plot_time_series, plot_frequency_spectrum
from core.feature_analysis import analyze_global_features

logger = logging.getLogger(__name__)


def build_tools(cleaned_df):
    """
    Build LangChain tools that close over (capture) cleaned_df.
    This avoids globals and lets the chatbot pass the DF into tool calls implicitly.
    """

    # -----------------------
    # Plot tool
    # -----------------------
    def plot_wrapper(query: str) -> str:
        logger.info(f"🎨 PLOT TOOL CALLED with query: '{query}'")
        start_time = datetime.now()

        try:
            parts = query.lower().split()
            sensor = parts[0]
            belt_status = parts[1].upper()
            domain = parts[2]
            condition = parts[3] if len(parts) > 3 else "vel-fissa"
            extra = parts[4] if len(parts) > 4 else None

            # Pass DF through to plotting functions (you must update these functions to accept df)
            if domain == "time":
                fig = plot_time_series(
                    df=cleaned_df,
                    sensor=sensor,
                    condition=condition,
                    belt_status=belt_status,
                    stwin=extra,
                )
            else:
                fig = plot_frequency_spectrum(
                    df=cleaned_df,
                    sensor=sensor,
                    condition=condition,
                    belt_status=belt_status,
                    stwin=extra,
                )

            st.pyplot(fig)

            elapsed = (datetime.now() - start_time).total_seconds()
            return f"✅ Plotted {sensor} ({belt_status}) in {condition} [{domain}]. (took {elapsed:.2f}s)"

        except Exception as e:
            logger.exception("Plotting error")
            return f"❌ Plotting error: {e}"

    plot_tool = Tool(
        name="PlotSensor",
        func=plot_wrapper,
        description=(
            "Plot sensor signals. Format: "
            "'<sensor> <OK/KO> <time/frequency> [condition] [stwin/pmi]'. "
            "Examples: 'mic OK time', 'acc KO frequency vel-fissa'"
        ),
    )

    # -----------------------
    # Feature importance tool
    # -----------------------
    def feature_wrapper(_: str = "") -> str:
        logger.info("⭐ FEATURE TOOL CALLED")
        start_time = datetime.now()

        try:
            # Pass DF into analysis function (you must update it to accept df)
            out = analyze_global_features(df=cleaned_df)

            top = out["top_features"][:10]
            st.write("### Top 10 Features by Discriminative Power:")
            st.table(top)

            elapsed = (datetime.now() - start_time).total_seconds()
            return f"✅ Displayed top 10 discriminative features. (took {elapsed:.2f}s)"

        except Exception as e:
            logger.exception("Feature analysis error")
            return f"❌ Feature analysis error: {e}"

    feature_tool = Tool(
        name="FeatureImportance",
        func=feature_wrapper,
        description="Analyzes and displays the top 10 most discriminative features between OK and KO samples.",
    )

    def inspect_dataset_wrapper(query: str = "") -> str:
        """Provides information about the sensor dataset."""
        try:
            info = []
            info.append(f"Dataset Shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns\n")
            
            # Column information
            info.append("Available Columns:")
            for col in cleaned_df.columns:
                info.append(f"  - {col}")
            
            # Label distribution
            if 'belt_status' in cleaned_df.columns:
                label_counts = cleaned_df['belt_status'].value_counts()
                info.append(f"\nBelt Status Distribution:")
                for status, count in label_counts.items():
                    info.append(f"  - {status}: {count} samples")
            
            # Sensor types
            if 'sensor' in cleaned_df.columns:
                sensors = cleaned_df['sensor'].unique()
                info.append(f"\nAvailable Sensors: {', '.join(sensors)}")
            
            # Conditions
            if 'condition' in cleaned_df.columns:
                conditions = cleaned_df['condition'].unique()
                info.append(f"\nConditions: {', '.join(conditions)}")
            
            return "\n".join(info)
        except Exception as e:
            return f"Error inspecting dataset: {e}"

    inspect_tool = Tool(
        name="InspectDataset",
        func=inspect_dataset_wrapper,
        description="Shows dataset structure, available sensors, conditions, and label distribution. Use this first to understand the data."
    )


    return [plot_tool, feature_tool, inspect_tool]
