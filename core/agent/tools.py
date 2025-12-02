from langchain.tools import Tool
from core.plotting import plot_time_series, plot_frequency_spectrum
from core.feature_analysis import analyze_global_features
import streamlit as st

def plot_wrapper(query: str) -> str:
    try:
        parts = query.lower().split()
        sensor, condition, domain = parts[0], parts[1].upper(), parts[2]
        if domain == "time":
            fig = plot_time_series(sensor, condition)
        else:
            fig = plot_frequency_spectrum(sensor, condition)
        st.pyplot(fig)
        return f"Plotted {domain} domain for {sensor} sensor in {condition} condition."
    except Exception:
        return "Please specify like 'mic KO time'."
    
plot_tools = [
    Tool(
        name="PlotSensor",
        func=plot_wrapper,
        description="Use to plot a sensor. Input format: '<sensor> <OK/KO> <time/frequency>'"
    )
]


def feature_importance_tool(_):
    top = analyze_global_features().head(10)
    st.write("Top 10 Features by Discriminative Power:")
    st.table(top)
    return "Displayed top 10 discriminative features."

feature_tools = [
    Tool(
        name="FeatureImportance",
        func=feature_importance_tool,
        description="Displays the most discriminative features that distinguish OK and KO samples."
    )
]

