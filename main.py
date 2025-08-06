import os
import time
import re
import numpy as np
import subprocess
import threading
import ast
import pprint
from datetime import datetime
from typing import TypedDict, Literal, List, Tuple, Dict

from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Configuration & Constants ---
EMB_VALUES = [384, 768, 1024, 2048, 4096]
LR_VALUES = [round(lr, 5) for lr in np.linspace(0.005, 0.0000005, 100).tolist()]
TOTAL_SEARCH_SPACE = len(EMB_VALUES) * len(LR_VALUES)
SCORE_THRESHOLD = 185 # Stop when the best score reaches this value
MAX_TRIALS = 3

def log_message(agent_id: int, message: str):
    """Prints a message with a timestamp and agent ID."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp} | Agent-{agent_id:02d}] {message}")

# --- LLM and Retry Wrapper ---
@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=20, min=20, max=120),
    retry=retry_if_exception_type(Exception))
def safe_llm_call(agent_id: int, prompt: str) -> str:
    log_message(agent_id, "📤 Calling LLM...")
    start_time = time.time()
    response = llm.invoke(prompt)
    end_time = time.time()
    duration = end_time - start_time
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", None)
    completion_tokens = usage.get("completion_tokens", None)
    total_tokens = usage.get("total_tokens", None)

    if total_tokens:
        print(f" Time: {duration:.2f}s | Tokens: {total_tokens} | Throughput: {total_tokens / duration:.2f} tokens/sec")
    else:
        print(f" Time: {duration:.2f}s (Token usage info not available)")

    return response.content.strip()

# --- Utility Functions ---
def key_from_params(emb: int, lr: float) -> str: return f"{emb}_{lr:.5f}"
def format_lr_string(lr: float) -> str: return f"{int(round(lr * 100000))}"

def parse_llm_response(response: str) -> List[Tuple[int, float]]:
    try:
        start, end = response.rfind("["), response.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No complete [...] block found in LLM response.")
        cleaned = response[start:end + 1]
        parsed = ast.literal_eval(cleaned)
        clean_list = []
        for i, item in enumerate(parsed):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                e, l = item
                clean_list.append((int(e), float(l)))
            else:
                raise ValueError(f"Invalid item at index {i}: {item}")
        return clean_list
    except Exception as e:
        print(f"❌ Failed to parse LLM response: {e}")
        return []

# --- Core Job Submission and Monitoring Logic ---
def submit_job(emb: int, lr: float) -> str:
    lr_str = format_lr_string(lr)
    sbatch_cmd = ["sbatch", "/lustre/orion/stf218/proj-shared/brave/climate-vit/submit_frontier.sh", str(emb), str(lr), lr_str]
    try:
        output = subprocess.check_output(sbatch_cmd, stderr=subprocess.STDOUT).decode()
        match = re.search(r"Submitted batch job (\d+)", output)
        return match.group(1) if match else None
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Job submission failed:\n{e.output.decode()}")
        return None

def wait_for_job_completion(agent_id: int, job_id: str):
    """Waits for a job to disappear from the squeue."""
    log_message(agent_id, f"⏳ Waiting for Slurm job {job_id} to disappear from the queue...")
    while True:
        try:
            output = subprocess.check_output(
                ["squeue", "--noheader", "--job", job_id],
                stderr=subprocess.STDOUT
            ).decode()
            if not output.strip():
                break # Job is gone.
        except subprocess.CalledProcessError:
            break # Job is gone.
        
        time.sleep(30)
    log_message(agent_id, f"✅ Job {job_id} is no longer in squeue.")

def extract_result_metrics(logfile_path: str):
    samples_sec, train_loss = None, None
    if not os.path.exists(logfile_path):
        return None, None
    with open(logfile_path, 'r') as f:
        for line in f:
            if m := re.search(r"avg samples/sec=(\d+\.\d+)", line): samples_sec = float(m.group(1))
            if m := re.search(r"Avg train loss=(\d+\.\d+)", line): train_loss = float(m.group(1))
    return samples_sec, train_loss

# --- Multi-Agent Orchestration ---
def get_next_job_params(agent_id: int, shared_state: Dict, lock: threading.Lock) -> Tuple[int, float] | None:
    with lock:
        if shared_state.get('stop_signal', False) or len(shared_state['tried']) >= TOTAL_SEARCH_SPACE:
            return None
        tried_summary = "\n".join([f"({k.replace('_', ', ')}): {v:.4f}" for k,v in shared_state['tried'].items() if isinstance(v, float)])
    
    prompt = f"""
You are an AI assistant for hyperparameter tuning. Your objective is to find the best (embedding, learning rate) configuration that **MAXIMIZES** a score.
### Current status:
- Total combinations possible: {TOTAL_SEARCH_SPACE}
- Combinations evaluated so far: {len(shared_state['tried'])}
- Previously tried combinations and their scores:
{tried_summary if tried_summary else "    None yet."}
### Your Task:
Based on the results so far, select up to **5** new, promising, untried `(embedding, learning_rate)` pairs to test next.
- **Available Embedding Sizes**: {EMB_VALUES}
- **Available Learning Rates**: A range from {LR_VALUES[0]} to {LR_VALUES[-1]}
Respond with a Python list of tuples, like: `[(1024, 0.00123), (2048, 0.00050)]`
It should be the last thing you write.
"""
    try:
        response = safe_llm_call(agent_id, prompt)
        suggested_params = parse_llm_response(response)
    except Exception as e:
        log_message(agent_id, f"❌ An error occurred during LLM call: {e}")
        return None
    
    if not suggested_params:
        log_message(agent_id, "LLM returned no new valid parameters.")
        return None
     
    with lock:
        num_tried = len(shared_state['tried'])
        if shared_state.get('stop_signal', False) or num_tried >= TOTAL_SEARCH_SPACE or num_tried >= MAX_TRIALS:
            return None
        for emb, lr in suggested_params:
            key = key_from_params(emb, lr)
            if key not in shared_state['tried']:
                log_message(agent_id, f"🔐 [LOCKED] Claiming job ({emb}, {lr:.5f}).")
                shared_state['tried'][key] = "running"
                return emb, lr
    
    log_message(agent_id, "All LLM suggestions were already taken by other agents.")
    return None

def agent_worker(agent_id: int, shared_state: Dict, lock: threading.Lock):
    log_message(agent_id, "🚀 Agent initialized and starting main loop.")
    while True:
        params = get_next_job_params(agent_id, shared_state, lock)
        if params is None:
            with lock:
                num_tried = len(shared_state['tried'])
                if shared_state.get('stop_signal', False) or num_tried >= TOTAL_SEARCH_SPACE or num_tried >= MAX_TRIALS:
                    if shared_state.get('stop_signal', False):
                        log_message(agent_id, "🏁 Stop signal received. Shutting down.")
                    elif num_tried >= MAX_TRIALS:
                        log_message(agent_id, f"🏁 Max trials limit ({MAX_TRIALS}) reached. Shutting down.")
                    else:
                        log_message(agent_id, "🏁 Search space exhausted. Shutting down.")
                    break
                
            log_message(agent_id, "💤 No new jobs available. Resting for 60s.")
            time.sleep(60)
            continue
        
        emb, lr = params
        key = key_from_params(emb, lr)
        job_id = submit_job(emb, lr)

        if not job_id:
            log_message(agent_id, f"❌ Job submission failed. Releasing claim.")
            with lock:
                del shared_state['tried'][key]
            continue

        wait_for_job_completion(agent_id, job_id)

        lr_str = format_lr_string(lr)
        log_path = f"/lustre/orion/stf218/proj-shared/brave/climate-vit/result/log.emb{emb}.lr{lr_str}"

        # If log file does not exist, release the claim and get a new job.
        if not os.path.exists(log_path):
            log_message(agent_id, f"‼️ Log file not found for job {job_id}. Releasing claim and getting new job.")
            with lock:
                del shared_state['tried'][key] # "Pretend it never happened"
            continue # Immediately loop to get a new job.

        metrics = extract_result_metrics(log_path)
        score = 0.0
        if metrics:
            samples, loss = metrics
            if samples is not None and loss is not None and loss > 0:
                 score = samples / loss
        
        with lock:
            log_message(agent_id, f"📝 [LOCKED] Updating shared state with score {score:.4f} for {key}.")
            shared_state['tried'][key] = score
            print(f"--- 'tried' dictionary updated by Agent-{agent_id:02d} ---")
            print(shared_state['tried'])
            if score > shared_state['best_score']:
                log_message(agent_id, f"🎉 New best score found! {score:.4f}")
                shared_state['best_score'] = score
                shared_state['best_params'] = (emb, lr)
                if shared_state['best_score'] >= SCORE_THRESHOLD:
                    log_message(agent_id, f"🏆 SCORE THRESHOLD REACHED! Signaling all agents to stop.")
                    shared_state['stop_signal'] = True

def monitor_progress(shared_state: Dict, lock: threading.Lock, stop_event: threading.Event, interval: int = 300):
    while not stop_event.is_set():
        time.sleep(interval)
        with lock:
            if shared_state.get('stop_signal', False):
                print("\n" + "="*20 + " MONITOR UPDATE (STOP SIGNALLED) " + "="*20)
            else:
                print("\n" + "="*20 + " MONITOR UPDATE " + "="*20)
            
            tried_count = len([v for v in shared_state['tried'].values() if isinstance(v, float)])
            running_count = len(shared_state['tried']) - tried_count
            best_score = shared_state['best_score']
            best_params = shared_state['best_params']
            
            print(f"Progress: {tried_count} completed, {running_count} running ({len(shared_state['tried'])} total claimed) / {TOTAL_SEARCH_SPACE}.")
            if best_score > 0:
                print(f"Current Best Score: {best_score:.4f} (Threshold: {SCORE_THRESHOLD})")
                print(f"Current Best Params: (emb={best_params[0]}, lr={best_params[1]:.5f})")
            else:
                print("No successful runs completed yet.")
            print("="*65 + "\n")

# --- Main Execution ---
if __name__ == "__main__":
    print("==== Multi-Agent Slurm Auto-Tuner ====")
    NUM_AGENTS = 1

    try:
        with open('.nemotron_token', 'r') as token_file: TOKEN = token_file.readline().strip()
        llm = ChatOpenAI(api_key=f"{TOKEN}", base_url="https://obsidian.ccs.ornl.gov/ai/nemotron/api/v1", model_name="Llama-3_1-Nemotron-Ultra-253B-v1-FP8")
    except FileNotFoundError:
        print("❌ Token file '.nemotron_token' not found. Exiting.")
        exit(1)

    shared_state = {
        "tried": {},
        "best_score": 0.0,
        "best_params": (0, 0.0),
        "stop_signal": False
    }
    state_lock = threading.Lock()
    
    monitor_stop_event = threading.Event()
    monitor = threading.Thread(target=monitor_progress, args=(shared_state, state_lock, monitor_stop_event))
    monitor.start()

    threads = []
    start_time = time.time()

    for i in range(1, NUM_AGENTS + 1):
        thread = threading.Thread(target=agent_worker, args=(i, shared_state, state_lock))
        threads.append(thread)
        thread.start()
        time.sleep(0.5)

    for thread in threads:
        thread.join()
    
    end_time = time.time()
    total_duration = end_time - start_time
   
    print("\n==== All agent threads have completed. Finalizing. ====")
    monitor_stop_event.set()
    monitor.join()

    # --- Print final results to console ---
    print("\n" + "="*20 + " FINAL RESULTS " + "="*20)
    print(f"📈 Total Execution Time: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")
    print(f"🏆 Best Score Achieved: {shared_state['best_score']:.4f}")
    print(f"🎯 Best Parameters: emb={shared_state['best_params'][0]}, lr={shared_state['best_params'][1]:.5f}")
    print("\n" + "="*20 + " FULL 'TRIED' DICTIONARY " + "="*20)
    pprint.pprint(shared_state['tried'])

    # --- Write final results to a file ---
    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    results_filename = f"tuning_results_{timestamp_str}.txt"
    try:
        with open(results_filename, 'w') as f:
            f.write("==== Hyperparameter Tuning Final Results ====\n")
            f.write(f"Agents Used: {NUM_AGENTS}\n")
            f.write(f"Run completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*45 + "\n\n")
            f.write(f"Total Execution Time: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)\n")
            f.write(f"Best Score Achieved: {shared_state['best_score']:.4f}\n")
            f.write(f"Best Parameters: emb={shared_state['best_params'][0]}, lr={shared_state['best_params'][1]:.5f}\n\n")
            f.write("==== Full 'Tried' Dictionary ====\n")
            f.write(pprint.pformat(shared_state['tried']))

        print("\n" + "="*58)
        print(f"✅ Results successfully saved to file: {results_filename}")
        print("="*58)
    except IOError as e:
        print(f"\n❌ Error saving results to file: {e}")