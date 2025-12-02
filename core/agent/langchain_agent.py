# agent/agent_implementation.py

from langchain.agents import create_pandas_dataframe_agent, initialize_agent, AgentType
from langchain.llms import LlamaCpp  # or OpenAI/HuggingFaceHub if you prefer
from agent.tools import plot_tools, feature_tools
from agent.prompt_templates import get_custom_system_prompt

import pandas as pd

def build_combined_agent(path: str = "../feature_sets"):

    df_path = path + "/cleaned_df.csv"
    df = pd.read_csv(df_path)

    llm = LlamaCpp(
        model_path="./models/llama-2-7b.Q4_K_M.gguf",
        temperature=0,
        max_tokens=512,
        n_ctx=2048,
        verbose=False
    )
    #combine all tools
    tools = plot_tools + feature_tools
    #get custom prompt 
    custom_prompt = get_custom_system_prompt()

    # Combine tools + df access with the LLM
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        agent_executor_kwargs={
            "system_message": custom_prompt
        }
    )

    return agent


#strealit entry point
# from agent.langchain_agent import build_combined_agent
# import streamlit as st
# agent = build_combined_agent()

# st.title("Sensor Assistant")

# query = st.text_input("Ask about the data (e.g., 'show top features', 'plot acc KO time')")
# if query:
#     try:
#         response = agent.run(query)
#         st.write(response)
#     except Exception as e:
#         st.error(f"Agent error: {e}")
