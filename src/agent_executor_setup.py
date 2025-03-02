from langchain.agents import AgentExecutor
from typing import List

def create_agent_executor(agent, tools: List) -> AgentExecutor:
    """Creates an AgentExecutor instance using the provided agent and tools."""
    try:
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, 
            tools=tools, 
            verbose=True,
            return_intermediate_steps=True,
        )
        return agent_executor
    except Exception as e:
        print(f"Error creating agent executor: {e}")
        return None