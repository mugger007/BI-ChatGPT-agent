from langchain.chains.llm import LLMChain

# Create LLM chain consisting of the LLM and a prompt
def create_llm_chain(llm, prompt):
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return llm_chain