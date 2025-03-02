from langchain.llms import GPT4All

local_path = "MODEL_FILE_PATH"

try:
    llm = GPT4All(model=local_path)
except Exception as e:
    print(f"Error loading LLM model: {e}")
    llm = None