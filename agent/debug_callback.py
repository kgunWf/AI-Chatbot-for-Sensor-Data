# agent/debug_callback.py

from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class DebugCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to track agent execution."""
    
    def __init__(self):
        self.tool_calls = []
        self.llm_calls = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs: Any) -> None:
        """Called when LLM starts."""
        logger.info("🤖 LLM STARTED")
        logger.info(f"  Prompt length: {len(prompts[0]) if prompts else 0} chars")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM ends."""
        logger.info("🤖 LLM ENDED")
        if hasattr(response, 'generations'):
            text = response.generations[0][0].text if response.generations else ""
            logger.info(f"  Response preview: {text[:200]}...")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Called when a tool starts."""
        tool_name = serialized.get("name", "Unknown")
        logger.info(f"🔧 TOOL STARTED: {tool_name}")
        logger.info(f"  Tool input: '{input_str}'")
        self.tool_calls.append({
            "tool": tool_name,
            "input": input_str,
            "status": "started"
        })
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool ends."""
        logger.info(f"🔧 TOOL ENDED")
        logger.info(f"  Tool output: {output[:200]}...")
        if self.tool_calls:
            self.tool_calls[-1]["status"] = "completed"
            self.tool_calls[-1]["output"] = output
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when a tool errors."""
        logger.error(f"🔧 TOOL ERROR: {error}")
        if self.tool_calls:
            self.tool_calls[-1]["status"] = "error"
            self.tool_calls[-1]["error"] = str(error)
    
    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Called when agent takes an action."""
        logger.info(f"🎯 AGENT ACTION: {action.tool}")
        logger.info(f"  Action input: {action.tool_input}")
    
    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        """Called when agent finishes."""
        logger.info(f"✅ AGENT FINISHED")
        logger.info(f"  Final output: {finish.return_values}")
    
    def get_summary(self) -> str:
        """Get a summary of all tool calls."""
        summary = f"\n{'='*60}\nTOOL CALL SUMMARY\n{'='*60}\n"
        summary += f"Total tool calls: {len(self.tool_calls)}\n\n"
        
        for i, call in enumerate(self.tool_calls, 1):
            summary += f"{i}. Tool: {call['tool']}\n"
            summary += f"   Input: {call['input']}\n"
            summary += f"   Status: {call['status']}\n"
            if 'output' in call:
                summary += f"   Output: {call['output'][:100]}...\n"
            if 'error' in call:
                summary += f"   Error: {call['error']}\n"
            summary += "\n"
        
        return summary