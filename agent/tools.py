# agent/tools.py
from langchain_core.tools import Tool
import logging
from datetime import datetime
import streamlit as st
import os
import sys
# Get the parent directory 
core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')) 
if core_dir not in sys.path: 
    sys.path.insert(0, core_dir)
from plotting import plotting
from feature_analysis import analyze_global_features

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
                # 5) Update Streamlit state with figures
                # -----------------------
                if "plots" not in st.session_state:
                    st.session_state.plots = []

                st.session_state.plots.extend(figures)
                
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

    def inspect_dataset_wrapper(_: str = "") -> str:
        """
        TERMINAL TOOL.
        Returns a concise, human-readable summary of the dataset.
        No analysis, no recommendations, no printing.
        """
        try:
            lines: list[str] = []

            # -----------------------------
            # Basic shape
            # -----------------------------
            n_rows, n_cols = cleaned_df.shape
            lines.append(f"**Dataset size:** {n_rows} samples × {n_cols} columns")

            # -----------------------------
            # Belt status distribution
            # -----------------------------
            if "belt_status" in cleaned_df.columns:
                counts = cleaned_df["belt_status"].value_counts()
                ok = int(counts.get("OK", 0))
                ko = int(counts.get("KO", 0))
                lines.append(f"**Belt status distribution:** OK = {ok}, KO = {ko}")

            # -----------------------------
            # Conditions
            # -----------------------------
            if "condition" in cleaned_df.columns:
                conditions = sorted(cleaned_df["condition"].dropna().unique())
                lines.append(f"**Conditions:** {', '.join(conditions)}")

            # -----------------------------
            # Sensors
            # -----------------------------
            if "sensor" in cleaned_df.columns:
                sensors = sorted(cleaned_df["sensor"].dropna().unique())
                lines.append(f"**Available sensors:** {', '.join(sensors)}")

            # -----------------------------
            # Columns (shortened, not overwhelming)
            # -----------------------------
            cols_preview = ", ".join(cleaned_df.columns[:12])
            suffix = " ..." if len(cleaned_df.columns) > 12 else ""
            lines.append(f"**Columns (preview):** {cols_preview}{suffix}")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ InspectDataset failed: {e}"


    inspect_tool = Tool(
        name="InspectDataset",
        func=inspect_dataset_wrapper,
        description="Shows dataset structure, available sensors, conditions, and label distribution. Use this first to understand the data."
    )


    return [plot_tool, feature_tool, inspect_tool]
