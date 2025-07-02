from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os

# Define state schema
class State(TypedDict):
    input: str
    llm_output:str
    final_output: str

os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"  # this is crucial

# vLLM-compatible model
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model_name="nemotron"   # Match --served-model-name
)

# LangGraph node
def llm_node(state: State) -> State:
    user_input = state["input"]
    result = llm.invoke(user_input)
    new_state= {**state, "llm_output": result.content} #creates a new dictionary, leaving state unchanged.
    return new_state 

def postprocess_node(state: State) -> State:
    response = state["llm_output"]
    summary = f"Length: {len(response)} characters"  # trivial example
    return {**state,  "final_output": summary}

# Build graph
graph = StateGraph(State)
graph.add_node("llm_response", llm_node)
graph.add_node("postprocess", postprocess_node)

graph.set_entry_point("llm_response")
graph.add_edge("llm_response", "postprocess")
graph.set_finish_point("postprocess")

# Compile and run
app = graph.compile()
output = app.invoke({"input": "What is LangGraph?"})
print(output["llm_output"])
print(output["final_output"])
