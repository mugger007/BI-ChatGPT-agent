from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool

def create_tools(df):
    tools = [
        Tool(
            name='python_repl',
            func=PythonAstREPLTool(locals={'df':df}),
            description="useful for running code and returns the output of the final line"
        )   
    ]
    return tools
