from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
    cache_dir="/lustre/orion/stf218/world-shared/palashmr/HF_backup/llama_253b",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
    cache_dir="/lustre/orion/stf218/world-shared/palashmr/HF_backup/llama_253b",
    trust_remote_code=True
)
