# agent/agent_implementation.py

import pandas as pd
from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from tools import build_tools

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
CSV_PATH = BASE_DIR / "core" / "output_dir" / "cleaned_df.csv"


class SensorDataAgent:
    """
    AI Agent for sensor data analysis using local Llama model.
    Provides plotting and feature analysis capabilities through natural language.
    """

    def __init__(self, df_path: str, model_path: str):
        self.df = pd.read_csv(df_path)
        
        # Initialize Llama model
        # self.llm = LlamaCpp(
        #     model_path=str(model_path),
        #     temperature=0.1,  # Low temperature for consistent outputs
        #     max_tokens=1024,
        #     n_ctx=4096,
        #     verbose=False,
        #     n_batch=512,
        #     top_p=0.9,
        # )
        from langchain_community.chat_models import ChatOllama
        self.llm = ChatOllama(
            model="phi3:mini",
            temperature=0,
        )

        # Create all tools
        self.tools = self._create_tools()
        
        # Create agent executor
        self.agent_executor = self._create_agent()

    def _create_tools(self):
        """Combine all available tools with dataset inspection."""
        
        plot_tool, feature_tool, inspect_tool = build_tools(self.df)

        return [plot_tool, feature_tool, inspect_tool]

    def _create_agent(self):
        """Create ReAct agent with optimized prompt for Llama."""
        
        # Simplified ReAct prompt optimized for Llama models
        template = """You are an AI assistant that helps analyze sensor data from a manufacturing belt system. The data contains OK (normal) and KO (faulty) samples.

Available Tools:
{tools}

Tool Names: {tool_names}

Follow this exact format:

Question: the user's question
Thought: think step by step about what to do
Action: choose ONE tool from [{tool_names}]
Action Input: the input for that tool
Observation: the tool's result
... (you can repeat Thought/Action/Action Input/Observation multiple times)
Thought: I now have enough information to answer
Final Answer: your complete answer to the user

Guidelines:
1. For plotting: Use format '<sensor> <OK/KO> <time/frequency> [condition]'
   - Valid sensors: acc, gyro, mag, mic, press, temp, hum
   - Example: "acc OK time" or "mic KO frequency vel-fissa"
   
2. For feature analysis: Just use "FeatureImportance" with empty input

3. Start with InspectDataset if you need to understand the data structure

4. Be concise and clear in your Final Answer

Current Question: {input}

{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,  # Set to False in production
            max_iterations=6,
            max_execution_time=60,  # Timeout after 60 seconds
            handle_parsing_errors="Check your output and make sure it follows the exact format with Action and Action Input!",
            return_intermediate_steps=False
        )

    def run(self, query: str, debug: bool = False) -> str:
        """Execute agent with user query."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"\n{'='*60}\nNEW QUERY: {query}\n{'='*60}")
        
        try:
            # Import debug callback if needed
            if debug:
                from debug_callback import DebugCallbackHandler
                callback = DebugCallbackHandler()
                result = self.agent_executor.invoke(
                    {"input": query},
                    config={"callbacks": [callback]}
                )
                logger.info(callback.get_summary())
            else:
                result = self.agent_executor.invoke({"input": query})
            
            output = result.get("output", "I couldn't process that request. Please try rephrasing.")
            logger.info(f"FINAL OUTPUT: {output}")
            return output
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"EXCEPTION: {error_msg}", exc_info=True)
            
            if "Could not parse LLM output" in error_msg:
                return "I had trouble understanding how to help with that. Could you rephrase your question?"
            return f"Error: {error_msg}"

    def get_dataset_summary(self) -> dict:
        """Get quick dataset statistics."""
        return {
            "total_samples": len(self.df),
            "columns": list(self.df.columns),
            "ok_samples": len(self.df[self.df['belt_status'] == 'OK']) if 'belt_status' in self.df.columns else 0,
            "ko_samples": len(self.df[self.df['belt_status'] == 'KO']) if 'belt_status' in self.df.columns else 0,
        }