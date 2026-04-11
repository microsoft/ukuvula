#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Production startup script for WhisperX ASR Pipeline with GPU acceleration
# This script sets up the environment and runs the pipeline with optimal settings.

set -euo pipefail
IFS=$'\n\t'

echo "🎙️  WhisperX ASR Pipeline - Production Startup"
printf '=%.0s' {1..60}
echo ""

# Detect conda environment path
ENV_NAME="nmf"
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --envs 2>/dev/null | grep "^${ENV_NAME} " | awk '{print $NF}')}"
if [[ -z "$CONDA_PREFIX" || ! -d "$CONDA_PREFIX" ]]; then
    echo "Error: Conda environment '${ENV_NAME}' not found. Run: conda activate ${ENV_NAME}"
    exit 1
fi

# Set up CUDA environment (conda-installed PyTorch places cuDNN in $CONDA_PREFIX/lib)
echo "Configuring CUDA environment..."
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_VISIBLE_DEVICES=0

# Ensure required arguments preserve spacing
MODEL_SIZE=${1:-"large-v2"}
INPUT_DIR_DEFAULT="data/nmf_recordings/Mandela at 90"
OUTPUT_DIR_DEFAULT="transcription_outputs"

if [[ $# -ge 2 ]]; then
    INPUT_DIR="$2"
else
    INPUT_DIR="$INPUT_DIR_DEFAULT"
fi

if [[ $# -ge 3 ]]; then
    RAW_OUTPUT_DIR="$3"
else
    RAW_OUTPUT_DIR="$OUTPUT_DIR_DEFAULT"
fi

# Capture any additional flags after the first three positional arguments and forward them to Python unchanged
shift $(( $# > 0 ? 1 : 0 )) || true
if [[ $# -gt 0 ]]; then shift || true; fi
if [[ $# -gt 0 ]]; then shift || true; fi
EXTRA_ARGS=("$@")

# If user passed a base results dir (e.g. 'results'), append basename of input root (e.g. 'nmf_recordings')
INPUT_BASENAME=$(basename "${INPUT_DIR%/}")
if [[ "$RAW_OUTPUT_DIR" == "results" || "$RAW_OUTPUT_DIR" == "results/" ]]; then
    OUTPUT_DIR="$RAW_OUTPUT_DIR/$INPUT_BASENAME"
else
    OUTPUT_DIR="$RAW_OUTPUT_DIR"
fi

if [[ -f "$INPUT_DIR" ]]; then
    echo "❌ The input path points to a file. Please provide a directory that contains audio files."
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "❌ Input directory does not exist: $INPUT_DIR"
    echo "💡 Remember to wrap paths containing spaces in quotes."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "CUDA environment configured"

echo "Configuration:"
echo "  Model: $MODEL_SIZE"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Conda env: $CONDA_PREFIX"

# Verify GPU is working
echo ""
echo "GPU Status:"
"$CONDA_PREFIX/bin/python" -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('GPU Not Available - will use CPU')
"

echo ""
echo "Starting WhisperX ASR Pipeline..."
echo "Press Ctrl+C to stop processing"
echo ""

# Run the pipeline with optimal GPU settings
"$CONDA_PREFIX/bin/python" src/pipeline/create_transcription_main.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_size "$MODEL_SIZE" \
    --use_gpu true \
    --language en \
    --enable_diarization false \
    --vad_device cuda \
    --min_confidence 0.4 \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Processing completed!"
echo "Results saved to: $OUTPUT_DIR"