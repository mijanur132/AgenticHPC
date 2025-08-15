# langchain_test.py
# ------------------------------------------------------------
# 1)  pip install langchain-openai  (once, inside your conda env)
# ------------------------------------------------------------
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model_name="nemotron"   # Match --served-model-name
)

print(llm.invoke("Explain tensor parallelism in two sentences."))