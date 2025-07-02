###############################################################################
#RUN SALLOC FIRST............
#  Interactive compute node – modules & env already loaded                    #
###############################################################################
module load craype-accel-amd-gfx90a PrgEnv-gnu rocm/6.3.1 gcc/12.2.0
source /autofs/nccs-svm1_sw/frontier/python/3.10/miniforge3/23.11.0/etc/profile.d/conda.sh

conda activate vllm-rocm

# Stay fully offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Path that contains config.json
MODEL_DIR=/lustre/orion/stf218/world-shared/palashmr/HF_backup/nemotron4m

# Quick sanity check
test -f "$MODEL_DIR/config.json" || { echo "config.json not found"; exit 1; }

# Launch vLLM
python -m vllm.entrypoints.openai.api_server \
       --model "$MODEL_DIR" \
       --dtype bfloat16 \
       --tensor-parallel-size 8 \
       --max-model-len 131072 \
       --gpu-memory-utilization 0.85 \
       --trust-remote-code \
       --served-model-name nemotron

#vllm allows parallel attention??

