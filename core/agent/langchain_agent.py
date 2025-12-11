# agent/agent_implementation.py
from langchain.agents import initialize_agent
from langchain.llms import LlamaCpp
from agent.tools import plot_tools, feature_tools
from agent.prompt_templates import get_custom_system_prompt
import pandas as pd


class SensorDataAgent:
    """
    Combines:
    - DataFrame reasoning (LLM)
    - Plotting tools
    - Feature analysis tools
    """

    def __init__(self, df_path: str, model_path: str):
        self.df = pd.read_csv(df_path)

        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=0,
            max_tokens=512,
            n_ctx=2048,
            verbose=False
        )

        self.tools = plot_tools + feature_tools
        self.system_prompt = get_custom_system_prompt()

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            handle_parsing_errors=True,
            agent_executor_kwargs={"system_message": self.system_prompt}
        )

    def run(self, query: str):
        """
        Passes any user query to the agent.
        """
        return self.agent.run(query)

#Streamlit Entry Point
# streamlit_app.py
# import streamlit as st
# from agent.agent_implementation import SensorDataAgent

# st.title("Sensor Data Assistant")

# agent = SensorDataAgent(
#     df_path="../output_dir/cleaned_df.csv",
#     model_path="./models/llama-2-7b.Q4_K_M.gguf"
# )

# query = st.text_input(
#     "Ask something (e.g. 'plot acc KO time', 'show top features')"
# )

# if query:
#     try:
#         response = agent.run(query)
#         st.write(response)
#     except Exception as e:
#         st.error(f"Agent error: {e}")