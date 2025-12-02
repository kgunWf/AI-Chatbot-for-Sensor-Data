# agent/agent_implementation.py

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.llms import LlamaCpp  # or OpenAI/HuggingFaceHub if you prefer
from agent.command_parser import parse_command
from agent.prompt_templates import agent_system_message

# Import your actual business logic
from core.plotting import plot_time_series, plot_frequency_spectrum
from core.stats_engine import get_top_features

import matplotlib.pyplot as plt
import streamlit as st
import tempfile
import os

# 1. ✅ Load Local Model
llm = LlamaCpp(
    model_path="./models/llama-2-7b.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    verbose=False
)

# 2. ✅ Define LangChain-compatible Tools
def show_top_features(_):
    features = get_top_features().head(5)
    st.write("Top 5 Features by Class Separation:")
    st.table(features)
    return "Displayed feature ranking."

def plot_sensor(sensor: str, condition: str, domain: str):
    if domain == "time":
        fig = plot_time_series(sensor=sensor, condition=condition)
    else:
        fig = plot_frequency_spectrum(sensor=sensor, condition=condition)

    st.pyplot(fig)
    return f"Here is the {domain} domain plot for {sensor}, {condition} samples."

# To keep it LangChain-compatible, wrap args in a single input string
def plot_sensor_wrapper(query: str):
    # crude parsing from natural string like "mic KO time"
    try:
        parts = query.lower().split()
        sensor, condition, domain = parts[0], parts[1].upper(), parts[2]
        return plot_sensor(sensor, condition, domain)
    except Exception:
        return "Error: please specify like 'acc KO time'."

tools = [
    Tool(
        name="FeatureAnalysis",
        func=show_top_features,
        description="Use this to display the most important sensor features."
    ),
    Tool(
        name="PlotSensor",
        func=plot_sensor_wrapper,
        description="Use this to plot a sensor signal. Input format: '<sensor> <OK/KO> <time/frequency>'"
    ),
]

# 3. ✅ Initialize the LangChain Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4. ✅ Entry Point Called from Streamlit
def run_chat_response(user_input: str):
    try:
        response = agent.run(user_input)
        return response
    except Exception as e:
        return f"Agent failed: {str(e)}"
