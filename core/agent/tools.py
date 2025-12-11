# agent/tools.py
from langchain.tools import Tool
from core.plotting import plot_time_series, plot_frequency_spectrum
from core.feature_analysis import analyze_global_features
import streamlit as st

# -----------------------
# Plot tool
# -----------------------
def plot_wrapper(query: str) -> str:
    """
    Expected input format:
    '<sensor> <belt_status> <time/frequency> [condition] [pmi/stwin]'
    Examples:
      'mic OK time'
      'acc KO frequency vel-fissa'
      'mag OK time stwin_00001'
    """
    try:
        parts = query.lower().split()

        sensor = parts[0]
        belt_status = parts[1].upper()
        domain = parts[2]

        # Optional fields
        condition = parts[3] if len(parts) > 3 else "vel-fissa"
        extra = parts[4] if len(parts) > 4 else None

        if domain == "time":
            fig = plot_time_series(sensor=sensor,
                                   condition=condition,
                                   belt_status=belt_status,
                                   stwin=extra)
        else:
            fig = plot_frequency_spectrum(sensor=sensor,
                                          condition=condition,
                                          belt_status=belt_status,
                                          stwin=extra)

        st.pyplot(fig)
        return f"Plotted {sensor} ({belt_status}) in {condition} [{domain}]."

    except Exception as e:
        return f"Plotting error. Use: '<sensor> <OK/KO> <time/frequency> [condition]'. Error: {e}"


plot_tools = [
    Tool(
        name="PlotSensor",
        func=plot_wrapper,
        description=(
            "Plot sensor signals. Format: "
            "'<sensor> <OK/KO> <time/frequency> [condition] [stwin/pmi]'. "
            "Examples: 'mic OK time', 'acc KO frequency vel-fissa'"
        )
    )
]


# -----------------------
# Feature importance tool
# -----------------------
def feature_wrapper(query: str = "") -> str:
    """
    Runs the feature analysis and displays top 10 discriminative features.
    """
    try:
        top = analyze_global_features().head(10)
        st.write("Top 10 Features by Discriminative Power:")
        st.table(top)
        return "Displayed top 10 discriminative features."
    except Exception as e:
        return f"Feature analysis error: {e}"


feature_tools = [
    Tool(
        name="FeatureImportance",
        func=feature_wrapper,
        description="Displays the top 10 discriminative features between OK and KO samples."
    )
]
