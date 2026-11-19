#!/bin/bash
# One probe_teachers.py stage per job; array jobs shard over instructions (SLURM_ARRAY_TASK_ID).
#   sbatch -p gpushort --constraint=ampere --time=0:59:00 --array=0-5 probe.sbatch embed --model qwen3emb8b_q --nshards 6
#SBATCH --job-name=steer-probe
#SBATCH --partition=sae
#SBATCH --account=pilot_sae_gpu
#SBATCH --constraint=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/scratch/%u/steercond/probe2k/logs/%x-%A_%a.out
set -euo pipefail
export HF_HOME=/gpfs/scratch/acw749/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# same vLLM / Qwen3.6 environment as nCLAP/scripts/slurm/caption_restyle.sbatch
export LD_LIBRARY_PATH=/data/home/acw749/conda-envs/instruct_embed/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export CUDA_HOME=/share/apps/rocky9/spack/apps/linux-rocky9-x86_64_v4/none-none/cuda/13.1.1-aheqacb
export PATH=$CUDA_HOME/bin:/data/home/acw749/conda-envs/instruct_embed/bin:$PATH
cd /data/home/acw749/SteerCond
mkdir -p /gpfs/scratch/acw749/steercond/probe2k/logs
python probe_teachers.py "$@"
