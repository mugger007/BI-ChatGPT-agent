from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool
from typing import List

def create_tools(df) -> List[Tool]:
    """Creates a list of tools for the agent."""
    try:
        tools = [
            Tool(
                name='python_repl',
                func=PythonAstREPLTool(locals={'df': df}),
                description="Useful for running code and returns the output of the final line"
            )   
        ]
        return tools
    except Exception as e:
        print(f"Error creating tools: {e}")
        return []