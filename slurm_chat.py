import os
import subprocess
from typing import TypedDict, Literal, Union

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


class State(TypedDict):
    input: str
    response: str
    next: Literal["decision", "slurm_submit", "END"]


os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"  # vLLM or OpenRouter-compatible server

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model_name="nemotron"   # Match --served-model-name
)


def run_submit_script() -> str:
    try:
        output = subprocess.check_output([
            "sbatch", "/lustre/orion/stf218/proj-shared/brave/climate-vit/submit_frontier.sh"
        ], stderr=subprocess.STDOUT).decode()
        return f"Job submitted: {output.strip()}"
    except subprocess.CalledProcessError as e:
        return f"Submission error: {e.output.decode()}"

def slurm_submit_node(state: State) -> State:
    result = run_submit_script()
    return {
        "input": state["input"],
        "response": result,
        "next": "END"
    }

def decision_node(state: State) -> State:
    prompt = state["input"]
    result = llm.invoke(prompt).content

   
    if "submit" in prompt.lower():
        next_step = "slurm_submit"
    else:
        next_step = "END"

    return {
        "input": prompt,
        "response": result,
        "next": next_step
    }

# -----------------------
# 5. Build LangGraph
# -----------------------
graph = StateGraph(State)

graph.add_node("decision", decision_node)
graph.add_node("slurm_submit", slurm_submit_node)

graph.set_entry_point("decision")

graph.add_conditional_edges(
    "decision",
    lambda state: state["next"],
    {
        "slurm_submit": "slurm_submit",
        "END": END,
    }
)

graph.add_edge("slurm_submit", END)

# -----------------------
# 6. Run it
# -----------------------
app = graph.compile()


if __name__ == "__main__":
    print("==== Chatting with LangGraph Slurm agent ====\n")

    while True:
    # Trigger Slurm tool
        user_input = input ("User: ")
        if user_input.lower() in ["quit", "End"]:
            break
        result = app.invoke({"input": user_input, "response": "", "next": "decision"})
        print(">> Q:", result["input"])
        print(">> Agent:", result["response"])