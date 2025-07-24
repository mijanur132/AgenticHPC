import os
import time
import yaml
import re
import numpy as np
import subprocess
from typing import TypedDict, Literal, List, Tuple

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import langgraph.pregel
langgraph.pregel.RECURSION_LIMIT = 1000  # set before compile()


# ✅ Retry-safe wrapper for LLM call

@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=20, min=20, max=120),
    retry=retry_if_exception_type(Exception)
)

# def safe_llm_call(prompt: str) -> str:
#     print("📤 Sending prompt to LLM...")
#     return llm.invoke(prompt).content.strip()

def safe_llm_call(prompt: str) -> str:
    print("📤 Sending prompt to LLM...")
    start_time = time.time()
    response = llm.invoke(prompt)
    end_time = time.time()
    
    duration = end_time - start_time

    # Try to extract token usage if vLLM returns OpenAI-style usage
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", None)
    completion_tokens = usage.get("completion_tokens", None)
    total_tokens = usage.get("total_tokens", None)

    if total_tokens:
        print(f"⏱️ Time: {duration:.2f}s | Tokens: {total_tokens} | Throughput: {total_tokens / duration:.2f} tokens/sec")
    else:
        print(f"⏱️ Time: {duration:.2f}s (Token usage info not available)")

    return response.content.strip()



EMB_VALUES = [384, 768, 1024, 2048, 4096]
LR_VALUES = [round(lr, 5) for lr in np.linspace(0.005, 0.0000005, 100).tolist()]

def key_from_params(emb: int, lr: float) -> str:
    return f"{emb}_{lr:.5f}"

import ast
import re
from typing import List, Tuple



def parse_llm_response(response: str, tried: dict) -> List[Tuple[int, float]]:
    print("Raw LLM response:", repr(response))

    try:
        # Find the last bracketed block in the text
        start = response.rfind("[")
        end = response.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No complete [ ... ] block found.")

        cleaned = response[start:end + 1]
        print("Extracted bracketed content:", repr(cleaned))

        parsed = ast.literal_eval(cleaned)
        print("Parsed object:", parsed)
        print("Parsed type:", type(parsed))

        clean_list = []
        for i, item in enumerate(parsed):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                e, l = item
                clean_list.append((int(e), float(l)))
            else:
                raise ValueError(f"Invalid item at index {i}: {item}")

        return [
            (e, l)
            for e, l in clean_list
            if key_from_params(e, l) not in tried
        ]

    except Exception as e:
        print("Failed to parse LLM response:", str(e))
        return []

class State(TypedDict):
    input: str
    response: str
    next: Literal["decision", "slurm_submit", "extract_metrics", "END"]
    tried: dict[str, float]
    best_score: float
    best_params: Tuple[int, float]
    running_jobs: List[Tuple[str, int, float]]
    num_jobs: int
    stop_flag: bool

def generate_tuning_prompt(state: State) -> str:
    state["tried"] = {k: v for k, v in state["tried"].items() if np.isfinite(v)}
    tried_summary = "\n".join([
        f"    ({k.split('_')[0]}, {k.split('_')[1]}) → score = {v:.4f}"
        for k, v in state["tried"].items()
    ])
    #print(tried_summary)
    return f"""
            You are an AI assistant for hyperparameter tuning of a deep learning model. Your objective is to find the best (embedding, learning rate) configuration that **MAXIMIZES** the **score**.
            Each configuration that has already been evaluated is stored in a dictionary called `tried`, where:
            - The **key** is a string in the format `"embedding_learningrate"` (e.g., "768_0.00328"), representing one configuration.
            - The **value** is the **score** (a float) obtained for that configuration.
            - Parse the key by splitting it on the underscore `_` to extract the embedding (as `int`) and learning rate (as `float`).
            - Do **not** treat the key as an opaque string — it encodes the actual hyperparameters.
            - You are NOT allowed to try all possible combinations. You must intelligently choose only a subset — **no more than 50% of the total search space.**
            -🚫 Do NOT suggest any (embedding, learning rate) pair that has already been tried (i.e., exists in the `tried` dictionary).
            ### Current status:
            So far:
            • Tried combinations with observed score:{tried_summary}
            -Ignore any score with 0 value. Most likely that are starting dummy value. 
            Strategies to consider:
            - Start with a broad sweep across the space from the smallest model.smallest Learning Rate. Hihgest model highest LR. Then narrow down the search space 
            - Focus next on promising (emb, lr) regions with best score
            - Use smart tradeoffs to balance exploration and exploitation
            - Do not attempt to try all combinations
            Available:
            • Embedding sizes: {EMB_VALUES}
            • Learning rates: {[f"{lr:.5f}" for lr in LR_VALUES]}
            • Total combinations: {len(EMB_VALUES) * len(LR_VALUES)}
    
            Please select {state['num_jobs']} new untried (emb, lr) pairs from the available space to test next.
            Respond as a list of tuples, like: [(1024, 0.00123), (2048, 0.0005), ...]

            Only return a Python list of tuples. It should be the last thing you write.
            """


with open('.nemotron_token', 'r') as token:
        TOKEN = token.readline().strip()

llm = ChatOpenAI(
    api_key= f"{TOKEN}",
    base_url= "https://obsidian.ccs.ornl.gov/ai/nemotron/api/v1",
    model_name="Llama-3_1-Nemotron-Ultra-253B-v1-FP8"
)

# ================================
# 2. LLM + Nodes
# ================================
# llm = ChatOpenAI(
#     base_url="http://localhost:8000/v1",
#     api_key="dummy",
#     model_name="nemotron"
# )

def get_allocated_nodes() -> list[str]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        print("❌ SLURM_JOB_ID not set.")
        return []
    try:
        # Step 1: Get raw nodelist like frontier[06126,06266]
        nodelist = subprocess.check_output(
            ["squeue", "--noheader", "--format=%N", "--job", job_id],
            stderr=subprocess.STDOUT,
        ).decode().strip()

        # Step 2: Expand hostnames from nodelist
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist],
            stderr=subprocess.STDOUT,
        ).decode().strip().splitlines()

        print(f"✅ Compute nodes: {hostnames}")
        return hostnames
    except subprocess.CalledProcessError as e:
        print("❌ Error getting compute nodes:", e.output.decode())
        return []

ALL_NODES = get_allocated_nodes()
print(ALL_NODES)
COMPUTE_NODES = ALL_NODES[1:] if len(ALL_NODES) > 1 else []

def wait_for_file(path, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(path):
            return True
        time.sleep(2)
    return False

def run_submit_script() -> str:
    try:
        output = subprocess.check_output([
            "bash", "/lustre/orion/stf218/proj-shared/brave/climate-vit/submit_frontier.sh"
        ], stderr=subprocess.STDOUT).decode()
        return f"Job submitted: {output.strip()}"
    except subprocess.CalledProcessError as e:
        return f"Submission error: {e.output.decode()}"

def submit_job(emb: int, lr: float) -> str:
    args = f"--config=mp_emb{emb} --lr={lr:.8f}"
    log_dir = "./result"
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"log.emb{emb}.lr{format_lr_string(lr)}")
    lr_str = format_lr_string(lr)
    print(f"came to srun submit bash with {emb} and {lr}")
    srun_cmd = [ "sbatch", "/lustre/orion/stf218/proj-shared/brave/climate-vit/submit_frontier.sh", str(emb), str(lr), lr_str]
    try:
        output = subprocess.check_output(srun_cmd, stderr=subprocess.STDOUT).decode()
        print( f"Job submission output:\n{output}")
        job_id = None
        for line in output.splitlines():
            if "Submitted batch job" in line:
                job_id = line.strip().split()[-1]
                break
        return job_id

    except subprocess.CalledProcessError as e:
        print( f"Submission failed:\n{e.output.decode()}")
    return None

def extract_result_metrics(logfile_path):
    samples_sec = None
    train_loss = None
    print("logfile_path:", logfile_path)
    with open(logfile_path, 'r') as f:
        for line in f:
            match_samples = re.search(r"avg samples/sec=(\d+\.\d+)", line)
            if match_samples:
                samples_sec = float(match_samples.group(1))
            match_loss = re.search(r"Avg train loss=(\d+\.\d+)", line)
            if match_loss:
                train_loss = float(match_loss.group(1))
    return samples_sec, train_loss

def format_lr_string(lr: float) -> str:
    return f"{int(round(lr * 100000))}"


def decision_node(state: State) -> State:
    print("came to decision node-->EMB values", EMB_VALUES)
    tried = set(state["tried"].keys())
    available = [
        (e, l)
        for e in EMB_VALUES
        for l in LR_VALUES
        if key_from_params(e, l) not in tried
    ]
    tried = set(state["tried"].keys())
    total_param_count = len(EMB_VALUES) * len(LR_VALUES)
    if len(tried) >= total_param_count or state.get("stop_flag", False):
        return {
            **state,
            "running_jobs": [],
            "stop_flag": True,
            "next": "END"
        }

    tried_summary = "\n".join(
        f"({k.replace('_', ', ')}): {v:.5f}" for k, v in state["tried"].items()
    )
    prompt = generate_tuning_prompt(state)
    # response = llm.invoke(prompt).content.strip()
    
    try:
        response = safe_llm_call(prompt)
    except Exception as e:
        print(f"❌ LLM failed after retries: {e}")

    selected = parse_llm_response(response, state["tried"])
    if not selected:
        return {
            **state,
            "running_jobs": [],
            "stop_flag": False,
            "next": "decision"
        }
    print("compute nodes:", COMPUTE_NODES)
    compute_nodes = COMPUTE_NODES[:len(selected)]
    jobs = list(zip([e for e, _ in selected], [l for _, l in selected]))
    print("jobs:", jobs)

    return {
        **state,
        "running_jobs": jobs,
        "next": "slurm_submit"
    }

def slurm_submit_node(state: State) -> State:
    tried = dict(state["tried"])
    job_ids: List[str] = []

    for emb, lr in state["running_jobs"]:
        print(f">>> Launching job on slurm with emb={emb}, lr={lr}")
        job_id = submit_job(emb, lr)
        if job_id is not None:
            job_ids.append(job_id)
            tried[key_from_params(emb, lr)] = float("inf")
        else:
            print(f"Failed to launch job for emb={emb}, lr={lr}")

    if job_ids:
        print("⏳ Waiting for all submitted jobs to finish …")
    while job_ids:
        time.sleep(10)
        remaining: List[str] = []
        for jid in job_ids:
            try:
                out = subprocess.check_output(["squeue", "--noheader", "--job", jid], stderr=subprocess.STDOUT).decode()
                if jid in out:
                    remaining.append(jid)
            except subprocess.CalledProcessError:
                continue
        job_ids = remaining
    if not job_ids:
        print("✅ All submitted jobs have completed.")

    return {
        **state,
        "tried": tried,
        "next": "extract_metrics"
    }

def extract_metrics_node(state: State) -> State:
    best_score = state["best_score"]
    best_params = state["best_params"]
    updated = False
    tried = dict(state["tried"])
    print("log file extraction....")
    for  emb, lr in state["running_jobs"]:
        lr_str = format_lr_string(lr)
        log_path = os.path.abspath(f"/lustre/orion/stf218/proj-shared/brave/climate-vit/result/log.emb{emb}.lr{lr_str}")  
        print("log_path:", log_path)

        try:
            samples, loss = extract_result_metrics(log_path)
        except FileNotFoundError:
            print(f"Log file missing: {log_path}")
            samples, loss = None, None
        
        if samples and loss:
            score = samples / loss
            tried[key_from_params(emb, lr)] = score
            print(f">>> Score for emb={emb}, lr={lr:.5f}: {score:.5f}")
            # if score < best_score * 1.02:
            #     best_score = score
            #     best_params = (emb, lr)
            #     updated = True
        print("tried:", tried)
    stop_flag = updated
    return {
        **state,
        "tried": tried,
        "best_score": best_score,
        "best_params": best_params,
        "stop_flag": stop_flag,
        "next": "decision"
    }


graph = StateGraph(State)
graph.add_node("decision", decision_node)
graph.add_node("slurm_submit", slurm_submit_node)
graph.add_node("extract_metrics", extract_metrics_node)

graph.set_entry_point("decision")
# graph.add_conditional_edges("decision", lambda s: s["next"], {
#     "slurm_submit": "slurm_submit",
#     "END": END
# })
graph.add_conditional_edges("decision", lambda s: s["next"], {
    "slurm_submit": "slurm_submit",
    "decision": "decision",   # ✅ Add this
    "END": END
})
graph.add_edge("slurm_submit", "extract_metrics")
graph.add_edge("extract_metrics", "decision")

app = graph.compile()


if __name__ == "__main__":
    print("==== LangGraph Slurm Auto-Tuner Agent ====")
    user_input = "start tuning"
    state = {
        "input": user_input,
        "response": "",
        "next": "decision",
        "tried": {},
        "best_score": 00.0,
        "best_params": (0, 0.0),
        "running_jobs": [],
        "num_jobs": 10,  # Can be dynamic
        "stop_flag": False
    }

    #result = app.invoke(state)
    result = app.invoke(state, config={"recursion_limit": 1000})

    print("\nFinal Best Params:", result["best_params"])
    print("Best Score:", result["best_score"])
    print(tried)









    # srun_cmd = [
    #     "srun", "--nodes=1", "--ntasks=8",
    #     "--gpu-bind=closest", "-c7",
    #     "--network=disable_rdzv_get",
    #     "/usr/bin/env", "bash", "-l", "-c",
    #     f"""
    #     source export_DDP_vars.sh
    #     python -u train_mp.py {args}
    #     """
    # ]