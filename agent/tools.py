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
from core.plotting import plotting
from core.feature_analysis import analyze_global_features

logger = logging.getLogger(__name__)


def build_tools(cleaned_df, path_to_data: str = "") -> list[Tool]:
    """
    Build LangChain tools that close over (capture) cleaned_df.
    This avoids globals and lets the chatbot pass the DF into tool calls implicitly.
    """

    # -----------------------
    # Plot tool
    # -----------------------
    from datetime import datetime
    import streamlit as st
    import logging

    logger = logging.getLogger(__name__)


    def make_plot_wrapper(base_root: str):

        def plot_wrapper(query: str) -> str:
            logger.info(f"🎨 PLOT TOOL CALLED with query: '{query}'")
            start_time = datetime.now()

            try:
                # -----------------------
                # 1) Parse query
                # -----------------------
                parts = query.strip().split()

                if len(parts) < 3:
                    return (
                        "❌ Invalid plot command. Format:\n"
                        "<sensor|sensor_type> <OK/KO> <time|frequency> "
                        "[condition] [rpm|stwin]"
                    )

                sensor_or_type = parts[0].lower()
                belt_status = parts[1].upper()
                plot_type = parts[2].lower()

                condition = parts[3] if len(parts) > 3 else "vel-fissa"
                extra = parts[4] if len(parts) > 4 else None

                # -----------------------
                # 2) Decide sensor vs type
                # -----------------------
                sensor = None
                sensor_type = None

                known_types = {"acc", "gyro", "mag", "mic", "temp", "hum", "prs"}
                if sensor_or_type in known_types:
                    sensor_type = sensor_or_type
                else:
                    sensor = sensor_or_type

                # -----------------------
                # 3) Map extra → rpm / stwin
                # -----------------------
                rpm = None
                stwin = None

                if condition == "vel-fissa":
                    rpm = extra
                elif condition == "no-load-cycles":
                    stwin = extra

                # -----------------------
                # 4) Call RAW plotting logic
                # -----------------------
                figures = plotting(
                    base_root=base_root,
                    plot_type=plot_type,
                    sensor_type=sensor_type,
                    sensor=sensor,
                    belt_status=belt_status,
                    condition=condition,
                    rpm=rpm,
                    stwin=stwin,
                )

                if not figures:
                    return "⚠️ No plots were generated for the given parameters."

                # -----------------------
                # 5) Render figures in Streamlit
                # -----------------------
                for fig in figures:
                    st.pyplot(fig)

                elapsed = (datetime.now() - start_time).total_seconds()

                # -----------------------
                # 6) Return short tool response (agent-safe)
                # -----------------------
                return (
                    f"Plotted {plot_type} data for "
                    f"{sensor_type or sensor} "
                    f"({belt_status}, {condition}). "
                    f"({len(figures)} figure(s), {elapsed:.2f}s)"
                )

            except Exception as e:
                logger.exception("Plotting error")
                return f"❌ Plotting error: {e}"

        return plot_wrapper


    plot_tool = Tool(
    name="PlotSensor",
    func=make_plot_wrapper(base_root=path_to_data),
    description=(
        "Plot RAW sensor signals.\n\n"
        "Format:\n"
        "<sensor|sensor_type> <OK/KO> <time|frequency> "
        "[condition] [rpm|stwin]\n\n"
        "Examples:\n"
        "- mic OK time\n"
        "- acc KO frequency vel-fissa PMS_100rpm\n"
        "- iis3dwb_acc OK time no-load-cycles STWIN_00012"
        ),
    )

    # -----------------------
    # Feature importance tool
    # -----------------------
    def feature_wrapper(_: str = "") -> str:
        logger.info("⭐ FEATURE TOOL CALLED")
        start_time = datetime.now()

        try:
            out = analyze_global_features(df=cleaned_df)

            top = out["top_features"][:10]

            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info(f"✅ Feature analysis completed in {elapsed:.2f}s")
            return "\n".join(top)
            

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
