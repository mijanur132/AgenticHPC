# 🧠 LLM-Guided Hyperparameter Optimization on Frontier (ORNL)

This repository provides a framework for LLM-guided hyperparameter optimization on the Frontier supercomputer. It leverages `vLLM` to run large language models locally on compute nodes and uses LangGraph for agent-based decision-making. All jobs are managed via Slurm and designed to support asynchronous, multi-agent optimization workflows. However many of the packages are tested on Frontier and may have issue when tried on different machine. 

You need following stuffs to run this code base:
1. Access to a running LLM agent. In our case we have a DGX box running nemotron and GPT OSS model running. There are other ways of running LLM over vLLM. Ref: "https://docs.vllm.ai/en/latest/"
2. One code base which you can train using slurm. In our case it is "climate-vit" located in the experiment directory. The associated slurm script is "submit_frontier.sh". Our agent would call this "submit_frontier.sh" from main.py in agentic experiment folder. Replace with your code base and script.
3. This code base which handles the LLM based agents. "main.py" starts and controls the agents. 

contact #Junqi Yin (YINJ@ORNL.GOV)" for access or any other issues. 

---
### you can create the required environment using following yaml file:
conda env create -f environment.yml

### you can also use the "requirement.txt" file also.

### to access the nemotron and GPT-OSS models running on DGX boxes you need token access. If you are ORNL internal you can generate one using your own credentials and running this code. Else please contact #Junqi Yin (YINJ@ORNL.GOV)" for access or any issue. 



python token_dgx.py

### put the token in the experiment folder. 

### Now normally running main.py would start the automatic training agent working. However you also need your own ML model which you are trying to optimize. In our case that function is train_mp.py in the subfolder "climate-vit". 


### 🚀 To run experiments with SLURM execute "submit_frontier.sh"  which is inside "climate-vit". Make sure you can train your model using similar SLURM script yourself before trying with the agentic automation.
### To adapt this framework to your own experiment, you need to modify the following line in `main.py: submit_job' function

sbatch_cmd = ["sbatch", "climate-vit/submit_frontier.sh", str(emb), str(lr), lr_str]


### 🔧 Setting Up Your Experiment

### You can update your args and "train_mp.py" based on your experiment. Again, make sure you can run this in slurm standalone. You have to call this from the main.py using the sbatch_cmd string list above. 

### Once everything is set up you can main.py to start the agent based parameter search

python main.py

### you can change various parameters, search spaces, number of parallel agent inside the main.py


### ADDITIONAL:
## 🔧 Running Local vLLM on Frontier

To launch a local instance of `vLLM` (e.g., Nemotron) on a Frontier `salloc` node:

### 1. Get an `salloc` Node Request an interactive node from Frontier:

```bash
salloc -A <project_id> -N 1 -t 01:00:00 -p batch --gpus-per-node=1

### Run the provided startup script:

bash vllm_start.sh

### Open a Second Terminal and SSH into the Same Node. In a separate terminal, connect to the same interactive node:

ssh -Y <username>@<allocated_node>

### You now have two sessions: one running the vLLM server and another for interacting with it.
### Query the LLM via local_nemotron.py
### From the second terminal:

python local_nemotron.py


###This Python script demonstrates how to send prompts to the local LLM server using the OpenAI-compatible API an prints the model's responses.
