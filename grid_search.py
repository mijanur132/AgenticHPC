import os
import time
import subprocess
import re
import numpy as np
import csv

EMB_VALUES = [384, 768]#, 1024, 2048, 4096]
LR_VALUES = [round(lr, 5) for lr in np.linspace(0.005, 0.0000005, 100).tolist()]

def key_from_params(emb, lr):
    return f"{emb}_{lr:.5f}"

def format_lr_string(lr: float) -> str:
    return f"{int(round(lr * 100000))}"

def submit_job(emb: int, lr: float) -> str:
    args = f"--config=mp_emb{emb} --lr={lr:.8f}"
    log_dir = "./result"
    os.makedirs(log_dir, exist_ok=True)
    lr_str = format_lr_string(lr)
    print(f"Submitting job with emb={emb}, lr={lr}")

    srun_cmd = ["sbatch", "/lustre/orion/stf218/proj-shared/brave/climate-vit/submit_frontier.sh", str(emb), f"{lr:.8f}", lr_str]
    try:
        output = subprocess.check_output(srun_cmd, stderr=subprocess.STDOUT).decode()
        for line in output.splitlines():
            if "Submitted batch job" in line:
                return line.strip().split()[-1]
    except subprocess.CalledProcessError as e:
        print(f"Job submission failed: {e.output.decode()}")
    return None

def wait_for_jobs(job_ids):
    while job_ids:
        time.sleep(10)
        remaining = []
        for jid in job_ids:
            try:
                out = subprocess.check_output(["squeue", "--noheader", "--job", jid], stderr=subprocess.STDOUT).decode()
                if jid in out:
                    remaining.append(jid)
            except subprocess.CalledProcessError:
                continue
        job_ids = remaining
    print("✅ All jobs in batch finished.")

def extract_result_metrics(logfile_path):
    samples_sec = None
    train_loss = None
    try:
        with open(logfile_path, 'r') as f:
            for line in f:
                match_samples = re.search(r"avg samples/sec=(\d+\.\d+)", line)
                if match_samples:
                    samples_sec = float(match_samples.group(1))
                match_loss = re.search(r"Avg train loss=(\d+\.\d+)", line)
                if match_loss:
                    train_loss = float(match_loss.group(1))
    except FileNotFoundError:
        print(f"Log file not found: {logfile_path}")
    return samples_sec, train_loss

# ----------------------------
# Simple Grid Search Execution
# ----------------------------
all_params = [(e, l) for e in EMB_VALUES for l in LR_VALUES]
results = {}
csv_rows = []  # List of tuples (embedding, learning_rate, loss)
best_score = float("inf")
best_params = None
BATCH_SIZE = 10

for i in range(0, len(all_params), BATCH_SIZE):
    batch = all_params[i:i + BATCH_SIZE]
    job_ids = []
    param_map = {}

    for emb, lr in batch:
        job_id = submit_job(emb, lr)
        if job_id:
            job_ids.append(job_id)
            param_map[job_id] = (emb, lr)

    wait_for_jobs(job_ids)
    
    write_header = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["embedding", "learning_rate", "score"])

        for jid in job_ids:
            emb, lr = param_map[jid]
            lr_str = format_lr_string(lr)
            log_path = os.path.abspath(f"/lustre/orion/stf218/proj-shared/brave/climate-vit/result/log.emb{emb}.lr{lr_str}")
            print(f"Checking log: {log_path}")
            
            samples, loss = extract_result_metrics(log_path)
            print(f"Parsed metrics: samples/sec={samples}, loss={loss}")

            if samples is None or loss is None:
                print(f"⚠️ Skipping emb={emb}, lr={lr:.5f} due to missing or invalid log data.")
                continue  # Skip this iteration

            try:
                score = samples / loss
            except Exception as e:
                print(f"❌ Error computing score for emb={emb}, lr={lr:.5f}: {e}")
                continue

            results[key_from_params(emb, lr)] = score
            writer.writerow((emb, lr, score))  # Write result immediately
            print(f"✅ Recorded: emb={emb}, lr={lr:.5f}, score={score:.5f}")

            if score > best_score:
                best_score = score
                best_params = (emb, lr)
                print(f"🔥 New Best: emb={emb}, lr={lr:.5f}, loss={loss:.5f}")



print("\n🏁 Grid Search Complete")
print("📊 All Results:")
for k, v in sorted(results.items(), key=lambda x: x[1]):
    print(f"{k}: {v:.5f}")
print("\nResults written to results.csv")
