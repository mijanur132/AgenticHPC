# langchain_test.py
# ------------------------------------------------------------
# 1)  pip install langchain-openai  (once, inside your conda env)
# ------------------------------------------------------------
from langchain_openai import ChatOpenAI

# The model ID must match what /v1/models returned earlier.
MODEL_ID = "/lustre/orion/stf218/world-shared/palashmr/HF_backup/nemotron4m"
# or "nemotron4m" if you launched with --served-model-name nemotron4m

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",   # vLLM’s OpenAI-compatible endpoint
    api_key="dummy",                       # any string; vLLM ignores it
    model_name=MODEL_ID,
    # You can set temperature, max_tokens, etc. here too
)

print(llm.invoke("Explain tensor parallelism in two sentences."))
