#!/usr/bin/env bash
# Route C orbit LoRA (board #3738): Wan 2.2 Fun Control A14B, one expert per run (EXPERT=low, default: giro's last
# 10 of 20 steps; EXPERT=high: the first 10, where global lighting and shading are set; the cache serves both), trained with DiffSynth-Studio (Apache-2.0,
# vendor/diffsynth-studio, its own .venv) on clips from scripts/train/wan_orbit_data.py, in two stages:
#
#   wan_orbit_lora.sh cache DATA CACHE GPU                text and VAE encodings once (DiT not loaded)
#   wan_orbit_lora.sh train CACHE OUT GPUS MODE [args]    the LoRA on the cached inputs
#
# GPUS: "1" or "0,1" (DDP: each GPU holds the whole DiT, so not with MODE offload). MODE, the frozen DiT's
# precision: fp8 (fp8 storage, bf16 math: DiffSynth's low-VRAM recipe), int8 / fp8q (ComfyUI's int8 convrot /
# fp8 weights AND math, comfy-kitchen: the formats giro already runs), bf16, offload (bf16 streamed layer by
# layer, single GPU), int8-offload / fp8q-offload (the same, quantized math). FRAMES (default 81): the first FRAMES frames of each clip (4k+1). Extra args go to DiffSynth's train.py (e.g. --num_epochs 2 --save_steps 100).
# Conditioning as giro's GiroWanFunControlToVideo: frame 0 = the hero (input_image) = the reference image,
# the depth video as the control. Training loads only the DiT (the text and VAE encodings are cached). Caption: giro's PROXY_ORBIT_PROMPT (wan_orbit_data.py pack).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}  # fragmentation costs GBs at 14B + 36k tokens
cd "$(dirname "$0")/../../vendor/diffsynth-studio"
WAN=${WAN:-/mnt/t9/models/Wan2.2-Fun-A14B-Control}
EXPERT=${EXPERT:-low}  # which of Wan 2.2's two experts gets the LoRA: low (t ~0-900) or high (t ~900-1000)
DIT=$WAN/${EXPERT}_noise_model/diffusion_pytorch_model.safetensors
if [ "$EXPERT" = high ]; then bounds=(--max_timestep_boundary 0.358 --min_timestep_boundary 0)
else bounds=(--max_timestep_boundary 1 --min_timestep_boundary 0.358); fi
T5=$WAN/models_t5_umt5-xxl-enc-bf16.pth
VAE=$WAN/Wan2.1_VAE.pth
MODELS="[\"$DIT\",\"$T5\",\"$VAE\"]"
step=$1; shift
common=(--height 768 --width 576 --num_frames ${FRAMES:-81} --remove_prefix_in_ckpt pipe.dit. --lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2
        --lora_rank 32 --extra_inputs input_image,control_video,reference_image
        "${bounds[@]}")
case $step in
cache)
    data=$1 cache=$2 gpu=$3
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/accelerate launch --num_processes 1 --mixed_precision bf16 \
        examples/wanvideo/model_training/train.py "${common[@]}" \
        --model_paths "$MODELS" --tokenizer_path "$WAN/google/umt5-xxl" --dataset_base_path "$data" --dataset_metadata_path "$data/metadata.csv" \
        --data_file_keys video,control_video,reference_image --dataset_repeat 1 --dataset_num_workers 4 \
        --output_path "$cache" --task sft:data_process --offload_models "$DIT" --fp8_models "$T5" \
        --use_gradient_checkpointing --use_gradient_checkpointing_offload  # stored in the cache: sft:train reads them from there
    .venv/bin/python ../../scripts/train/wan_cache_fix.py "$cache"  # the hero as frame 0 in y, as giro's inference
    ;;
train)
    cache=$1 out=$2 gpus=$3 mode=$4; shift 4
    n=$(awk -F, '{print NF}' <<< "$gpus")
    prec=()
    case $mode in
    fp8) prec=(--fp8_models "$DIT") ;;
    int8) prec=(--quant_options "$DIT:comfy_kitchen_int8_w8a8") ;;
    fp8q) prec=(--quant_options "$DIT:comfy_kitchen_fp8_w8a8") ;;
    bf16) ;;
    offload) prec=(--enable_model_cpu_offload) ;;
    int8-offload) prec=(--quant_options "$DIT:comfy_kitchen_int8_w8a8" --enable_model_cpu_offload) ;;
    fp8q-offload) prec=(--quant_options "$DIT:comfy_kitchen_fp8_w8a8" --enable_model_cpu_offload) ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
    esac
    CUDA_VISIBLE_DEVICES=$gpus .venv/bin/accelerate launch --num_processes "$n" --mixed_precision bf16 \
        examples/wanvideo/model_training/train.py "${common[@]}" \
        --model_paths "[\"$DIT\"]" --dataset_base_path "$cache" --dataset_repeat 1 --output_path "$out" \
        --task sft:train --learning_rate 1e-4 --use_gradient_checkpointing \
        --use_gradient_checkpointing_offload --enable_csv_log "${prec[@]}" "$@"
    ;;
*) echo "usage: $0 cache DATA CACHE GPU | train CACHE OUT GPUS MODE [args]" >&2; exit 2 ;;
esac
