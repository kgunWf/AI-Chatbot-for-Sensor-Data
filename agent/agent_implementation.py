# agent/agent_implementation.py

import pandas as pd
from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatOllama
from tools import build_tools

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
CSV_PATH = BASE_DIR / "core" / "output_dir" / "cleaned_df.csv"
DATA_PATH = BASE_DIR / "data" / "Sensor_STWIN"

class SensorDataAgent:
    """
    AI Agent for sensor data analysis using local Llama model.
    Provides plotting and feature analysis capabilities through natural language.
    """

    def __init__(self, df_path: str, path_to_data: str):
        self.df = pd.read_csv(df_path)
        self.path_to_data = path_to_data
            # Initialize Llama model
        self.llm = ChatOllama(
            model="qwen2.5:7b-instruct",
            temperature=0,
            num_ctx=1024,
        )

        # Create all tools
        self.tools = self._create_tools()
        
        # Create agent executor
        self.agent_executor = self._create_agent()

    def _create_tools(self):
        """Combine all available tools with dataset inspection."""
        
        plot_tool, feature_tool, inspect_tool = build_tools(self.df, self.path_to_data)

        return [plot_tool, feature_tool, inspect_tool]

    def _create_agent(self):
        """Create ReAct agent with optimized prompt for Llama."""
        
        # Simplified ReAct prompt optimized for Llama models
        template = """You are an AI assistant that ROUTES user requests to tools for analyzing sensor data from a manufacturing belt system.
The dataset contains OK (normal) and KO (faulty) samples.

Your role is STRICTLY LIMITED:
- You route requests to tools.
- You DO NOT perform analysis yourself.
- You DO NOT design pipelines, reports, or workflows.

--------------------------------------------------
AVAILABLE TOOLS
--------------------------------------------------
{tools}

Tool names:
{tool_names}

You may ONLY use the tools listed above.
If you output an Action that is NOT in {tool_names}, the response is INVALID.

--------------------------------------------------
STRICT FORMAT (MANDATORY)
--------------------------------------------------
You MUST follow this format EXACTLY and COMPLETELY:

Question: the user's question
Thought: brief reasoning about which tool to use
Action: one tool name from {tool_names}
Action Input: the exact input for that tool (may be empty)
Observation: the tool's output
Thought: I now have enough information to answer.
Final Answer: a short, clear answer to the user.

Rules:
- Every Action MUST have an Action Input line (even if empty).
- Observation MUST appear exactly once.
- NOTHING is allowed after Final Answer.
- If this format is violated, the response is invalid.

--------------------------------------------------
TOOL ROUTING RULES (CRITICAL)
--------------------------------------------------

ALL TOOLS IN THIS SYSTEM ARE TERMINAL TOOLS.
Once a tool is called and its Observation is received:
- DO NOT call another tool
- DO NOT propose follow-up actions
- DO NOT refine, interpret, or extend the result
- IMMEDIATELY produce Final Answer

--------------------------------------------------
1) DATASET INSPECTION — InspectDataset (TERMINAL)
--------------------------------------------------

Use InspectDataset IF the user asks about:
- dataset structure
- available sensors
- available conditions
- columns
- number of samples
- OK / KO counts

Action: InspectDataset
Action Input: (empty)

After InspectDataset:
- DO NOT plot
- DO NOT analyze the feature importance
- DO NOT infer statistics
- ONLY summarize metadata in Final Answer
- Do not provide a general response, but give the specific dataset info requested

--------------------------------------------------
2) FEATURE IMPORTANCE — FeatureImportance (TERMINAL)
--------------------------------------------------

Use FeatureImportance IF the user asks about:
- top features
- feature importance
- discriminative features
- features separating OK vs KO

STRICT RULES FOR FeatureImportance:
- FeatureImportance takes NO parameters.
- Action Input MUST be empty.
- FeatureImportance is TERMINAL.

You are NOT allowed to:
- condition results on sensors
- condition results on belt status
- interpret statistics
- explain methodology
- suggest further analysis

The tool output MUST appear ONLY in Observation.
NEVER copy tool output into Action Input or Final Answer verbatim.

Correct example:

Question: Show top features
Thought: The user wants the most discriminative features between OK and KO.
Action: FeatureImportance
Action Input:
Observation:
acc_mean_z
acc_mean_y
temp_skew
acc_mean_x
gyro_mean_x
Thought: I now have enough information to answer.
Final Answer: Accelerometer- and temperature-based features are the most discriminative between OK and KO samples.

--------------------------------------------------
3) PLOTTING — PlotSensor (TERMINAL)
--------------------------------------------------

Use PlotSensor IF the user asks to:
- plot
- show
- visualize
- display
a signal or frequency spectrum.

Action Input MUST be EXACTLY ONE LINE in this format:
<sensor_or_type> <OK/KO> <time|frequency> [vel-fissa|no-load-cycles] [rpm|stwin]

Rules:
- sensor_or_type can be:
  - a sensor type: acc, gyro, mag, mic, temp, hum, prs
  - OR an exact sensor name (e.g., iis3dwb_acc)
- If condition is vel-fissa → optional last token is rpm (e.g., PMS_50rpm)
- If condition is no-load-cycles → optional last token is stwin (e.g., STWIN_00012)
- If the user says only "KO", pass "KO" (tool handles default mapping)
- DO NOT add extra words, punctuation, or explanations

Correct examples:
- acc OK time
- mic KO frequency
- gyro KO time vel-fissa PMS_100rpm
- iis3dwb_acc OK time no-load-cycles STWIN_00012

--------------------------------------------------
STRICT CONSTRAINTS (NON-NEGOTIABLE)
--------------------------------------------------

You are NOT allowed to:
- invent tools
- invent sensor names, conditions, rpm, or stwin values
- perform data analysis in text
- generate reports or essays
- write code
- propose machine learning workflows
- suggest plots beyond calling PlotSensor
- continue reasoning after a terminal tool

Action Input must contain ONLY the command string or be empty.

--------------------------------------------------
FINAL ANSWER RULES
--------------------------------------------------

- 1–3 sentences only
- Summarize what the tool did
- NO technical explanation
- NO methodology
- NO speculation

--------------------------------------------------
CURRENT QUESTION
--------------------------------------------------
{input}

{agent_scratchpad}

"""



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