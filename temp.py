# Save this as check_job.py
import subprocess
import sys

if len(sys.argv) != 2:
    print("Usage: python check_job.py <JOB_ID>")
    sys.exit(1)

job_id = sys.argv[1]
active_states = ["PENDING", "RUNNING", "PD", "R"]

print(f"--- Checking status for Job ID: {job_id} ---")

try:
    command = ["squeue", "--noheader", "--job", job_id, "--format=%T"]
    print(f"Running command: {' '.join(command)}")
    
    output = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT
    ).decode().strip()

    print(f"Raw output from squeue: '{output}'")
    print(f"Is job in an active state? ({output!r} in {active_states!r})")

    if output in active_states:
        print("Result: Job is considered ACTIVE. The agent would continue waiting.")
    else:
        print("Result: Job is considered FINISHED. The agent would proceed.")

except subprocess.CalledProcessError:
    print("\nCommand failed with CalledProcessError.")
    print("Result: Job is considered FINISHED. The agent would proceed.")