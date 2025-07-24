import os
import subprocess
from typing import TypedDict, Literal, Union

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import re
import numpy as np


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

def update_config_and_slurm(config_path, slurm_path, new_emb, new_lr):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['lr'] = new_lr
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    with open(slurm_path, 'r') as f:
        slurm_lines = f.readlines()
    updated_lines = []
    for line in slurm_lines:
        if line.strip().startswith("emb="):
            updated_lines.append(f"emb={new_emb}\n")
        else:
            updated_lines.append(line)
    with open(slurm_path, 'w') as f:
        f.writelines(updated_lines)
    print(f"Updated lr to {new_lr} in config")
    print(f"Updated emb={new_emb} in SLURM script")

def extract_result_metrics(logfile_path):
    samples_sec = None
    train_loss = None

    with open(logfile_path, 'r') as f:
        for line in f:
            # Match: avg samples/sec=32.928591
            match_samples = re.search(r"avg samples/sec=(\d+\.\d+)", line)
            if match_samples:
                samples_sec = float(match_samples.group(1))

            # Match: Avg train loss=0.625625
            match_loss = re.search(r"Avg train loss=(\d+\.\d+)", line)
            if match_loss:
                train_loss = float(match_loss.group(1))

    return samples_sec, train_loss

def run_submit_script() -> str:
    try:
        output = subprocess.check_output([
            "bash", "/lustre/orion/stf218/proj-shared/brave/climate-vit/submit_frontier.sh"
        ], stderr=subprocess.STDOUT).decode()
        return f"Job submitted: {output.strip()}"
    except subprocess.CalledProcessError as e:
        return19699199
         f"Submission error: {e.output.decode()}"

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
    #decision space
    embedding_list =[1024,2048,4096]
    lr_list = np.linspace(0.005, 0.00005, 100).tolist()
    # read the loss files and decide on model and lr combination
    # logfile = f'./result/log.emb{emb}.{lr}.n4'
    # samples_sec, train_loss = extract_final_metrics(logfile_path)
    #if condition met for breaking; stop trains
    #else write new slurm scripti and resubmit with new lr and embedding
    #
    
    # update_config_and_slurm(config_path, slurm_path, new_emb, new_lr):

   
    if "submit" in prompt.lower():
        next_step = "slurm_submit"
    else:
        next_step = "END"

    return {
        "input": prompt,
        "response": result,
        "next": next_step
    }


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