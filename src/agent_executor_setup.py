from langchain.agents import AgentExecutor

def create_agent_executor(agent, tools):
    # Creating an instance of the AgentExecutor class using the provided agent and tools.
    # The from_agent_and_tools method is used to create an instance.
    # The arguments include the agent, tools, verbose flag set to True,
    # return_intermediate_steps flag set to True, and memory is commented out.
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=True,
        return_intermediate_steps=True,
    )
    
    # Returning the created agent_executor instance.
    return agent_executor