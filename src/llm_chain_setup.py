from langchain.chains.llm import LLMChain
from typing import Any

def create_llm_chain(llm: Any, prompt: Any) -> LLMChain:
    """Creates an LLM chain consisting of the LLM and a prompt."""
    try:
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        return llm_chain
    except Exception as e:
        print(f"Error creating LLM chain: {e}")
        return None