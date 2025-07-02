# chat_vllm.py
# -----------------------------------------------------------
#  One-time install (if you haven’t already):
#     pip install -q langchain-openai
# -----------------------------------------------------------
from langchain_openai import ChatOpenAI

# ──── Configuration ─────────────────────────────────────────
BASE_URL  = "http://localhost:8000/v1"   # vLLM’s OpenAI endpoint
MODEL_ID  = "/lustre/orion/stf218/world-shared/palashmr/HF_backup/nemotron4m"
# If you launched with:  --served-model-name nemotron4m
#   then use:  MODEL_ID = "nemotron4m"
API_KEY   = "dummy"                      # any non-empty string
# ────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    base_url = BASE_URL,
    api_key  = API_KEY,
    model_name = MODEL_ID,
    temperature = 0.7,        # optional tweaks
)

print("Welcome to the Nemotron-8B chatbot! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        break
    # ChatOpenAI returns an AIMessage object; .content is the text
    response = llm.invoke(user_input)
    print("Bot:", response.content)
