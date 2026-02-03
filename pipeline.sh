#!/bin/bash
set -e

# ==========================================
# Configuration & Setup
# ==========================================

# Base Paths
BASE_DIR="./path/to/your/workspace"
TRAIN_DIR="$BASE_DIR/dictionary_learning_demo"
EVAL_DIR="$BASE_DIR/SAEBench"

# Environment Scripts
TRAIN_ENV="./path/to/your/train_env.sh"
EVAL_ENV="./path/to/your/eval_env.sh"

DATASET_NAME="openwebtext"

# Model to Layers Map
declare -A MODEL_LAYERS
MODEL_LAYERS["google/gemma-2-2b"]=12

# Helper variables for easy access
MODEL_2B="google/gemma-2-2b"
LAYERS_2B=${MODEL_LAYERS[$MODEL_2B]}
export WANDB_MODE="online"

TARGET_GB=70
TARGET_L0S="[60,80,100,120]"
AUX_LOSS_START_STEP=0
AUX_LOSS_INTERVAL=5
DECAY_START_FRACTION=0.8

if [ "$DATASET_NAME" == "openwebtext" ]; then
    DATASET_PATH="./path/to/your/datasets/openwebtext"
    DATASET_ARG="--dataset_path $DATASET_PATH"
    echo "Using OpenWebText dataset: $DATASET_PATH"
fi

EVAL_TYPES=(
    "absorption"
    "core"
    "autointerp"
    "ravel"
)
DIR_LEN=100
OPENAI_API_KEY="your-openai-api-key-here"
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_MODEL="gpt-5-mini"
OPENAI_MAX_WORKERS=30
RANDOM_SEED=42
REVERSE_FLAG="" # Default disabled

# Helper Functions
detect_dataset() {
    local sae_path="$1"
    if [[ "$sae_path" == *"fineweb"* ]]; then echo "fineweb"; elif [[ "$sae_path" == *"pile"* ]]; then echo "pile"; elif [[ "$sae_path" == *"openwebtext"* ]]; then echo "openwebtext"; else echo "fineweb"; fi
}

get_dataset_arg() {
    local dataset_name="$1"
    if [ "$dataset_name" == "openwebtext" ]; then
        local base_path="./path/to/your/datasets/openwebtext"
        echo "--dataset_path $base_path"
    else
        echo ""
    fi
}

get_gpu_memory_usage() {
    local gpu_id="$1"
    local memory_used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id")
    echo "scale=2; $memory_used_mb / 1024" | bc
}

wait_for_gpu_memory() {
    local gpu_id="$1"
    local needed_gb="$2"
    local total_memory=140
    echo "Waiting for GPU $gpu_id memory..."
    while true; do
        local current_usage=$(get_gpu_memory_usage "$gpu_id")
        local available_memory=$(echo "$total_memory - $current_usage" | bc -l)
        if (( $(echo "$available_memory >= $needed_gb" | bc -l) )); then
            echo "GPU $gpu_id memory ready."
            break
        fi
        sleep 5
    done
}

# ==========================================
# The Pipeline Commands
# ==========================================

commands=(
    # "python demo.py --save_dir trained_saes_icml_v11_500M_2048_${DATASET_NAME}_80_100_120 --model_name $MODEL_2B --layers $LAYERS_2B --sae_batch_size 2048 --architectures  top_k --device 'cuda:0' --use_wandb --wandb_project icml_v11_500M_${DATASET_NAME} --num_tokens 500000000 --target_gb $TARGET_GB --target_l0s '[80,100,120]'   --dtype 'float32' $DATASET_ARG"
    # "python demo.py --save_dir trained_saes_icml_v11_500M_2048_${DATASET_NAME}_80_100_120 --model_name $MODEL_2B --layers $LAYERS_2B --sae_batch_size 2048 --architectures  batch_top_k --device 'cuda:1' --use_wandb --wandb_project icml_v11_500M_${DATASET_NAME} --num_tokens 500000000 --target_gb $TARGET_GB --target_l0s '[80,100,120]'   --dtype 'float32' $DATASET_ARG"
    # "python demo.py --save_dir trained_saes_icml_v11_500M_2048_${DATASET_NAME}_80_100_120 --model_name $MODEL_2B --layers $LAYERS_2B --sae_batch_size 2048 --architectures  matryoshka_batch_top_k --device 'cuda:2' --use_wandb --wandb_project icml_v11_500M_${DATASET_NAME} --num_tokens 500000000 --target_gb $TARGET_GB --target_l0s '[80,100,120]'   --dtype 'float32' $DATASET_ARG"
    # "python demo.py --save_dir trained_saes_icml_v11_500M_2048_${DATASET_NAME}_80_100_120 --model_name $MODEL_2B --layers $LAYERS_2B --sae_batch_size 2048 --architectures  ort --device 'cuda:3' --use_wandb --wandb_project icml_v11_500M_${DATASET_NAME} --num_tokens 500000000 --target_gb $TARGET_GB --target_l0s '[80,100,120]'   --dtype 'float32' $DATASET_ARG --aux_loss_start_step 0 --aux_loss_interval 5"
    "python demo.py --save_dir trained_saes_icml_v11-6_500M_2048_${DATASET_NAME}_60 --model_name $MODEL_2B --layers $LAYERS_2B --sae_batch_size 2048 --architectures  batch_top_k --device 'cuda:4' --use_wandb --wandb_project icml_v11_500M_${DATASET_NAME} --num_tokens 500000000 --target_gb $TARGET_GB --target_l0s '[60]' --lambda_swc2r '[5]' --swc2r_alpha 1 --swc2r_tau 1.0 --swc2r_type "topTauPerFeatSquare" --dtype 'float32' $DATASET_ARG --aux_loss_start_step 0 --aux_loss_interval 5"
)

# ==========================================
# Execution Loop
# ==========================================

for i in "${!commands[@]}"; do
    # Run each iteration in background for parallel execution
    (
    CMD="${commands[i]}"
    echo "----------------------------------------------------------------"
    echo "Processing Pipeline Job $((i+1))"
    echo "----------------------------------------------------------------"

    # --- Parse Arguments ---
    # 1. Extract raw values from command string
    RAW_SAVE_DIR=$(echo "$CMD" | grep -oP "(?<=--save_dir )[^ ]+")
    MODEL_NAME=$(echo "$CMD" | grep -oP "(?<=--model_name )[^ ]+")
    LAYERS=$(echo "$CMD" | grep -oP "(?<=--layers )[^ ]+")
    DEVICE_RAW=$(echo "$CMD" | grep -oP "(?<=--device )['\"]?[^'\" ]+['\"]?")
    DEVICE=$(echo "$DEVICE_RAW" | tr -d "'\"")
    DTYPE=$(echo "$CMD" | grep -oP "(?<=--dtype )['\"]?[^'\" ]+['\"]?" | tr -d "'\"")

    # 2. Extract architectures (handle multiple space-separated values until next --)
    ARCHS_RAW=$(echo "$CMD" | grep -oP "(?<=--architectures )[^--]+")
    # Trim whitespace and join with _ (mimics '_'.join(args.architectures))
    ARCHS_JOINED=$(echo $ARCHS_RAW | xargs | tr ' ' '_')

    if [ -z "$RAW_SAVE_DIR" ] || [ -z "$MODEL_NAME" ] || [ -z "$LAYERS" ] || [ -z "$DEVICE" ] || [ -z "$ARCHS_JOINED" ]; then
        echo "Error: Could not parse required arguments from command."
        echo "Parsed: save_dir=$RAW_SAVE_DIR, model=$MODEL_NAME, layers=$LAYERS, archs=$ARCHS_JOINED, device=$DEVICE"
        exit 1
    fi

    # 3. Reconstruct ACTUAL_SAVE_DIR (mimics demo.py logic)
    # Logic: f"{args.save_dir}_{args.model_name}_{'_'.join(args.architectures)}".replace("/", "_")

    # Extract lambda params and construct suffix
    LAMBDA_SUFFIX=""

    process_lambda() {
        local suffix_name="$1"
        local arg_name="$2"

        # Extract value. Try to match quoted string first, then unquoted word.
        local val=$(echo "$CMD" | grep -oP "(?<=--$arg_name )['\"]?\[.*?\]['\"]?" || true)

        if [ -z "$val" ]; then
             val=$(echo "$CMD" | grep -oP "(?<=--$arg_name )[^ ]+" || true)
        fi

        # Clean up
        local clean_val=$(echo "$val" | tr -d "'\"[]")

        if [ -z "$clean_val" ]; then clean_val="0.0"; fi

        # Check for non-zero
        IFS=',' read -ra ADDR <<< "$clean_val"
        local first_val="${ADDR[0]}"
        local has_nonzero=false

        for v in "${ADDR[@]}"; do
            if (( $(echo "$v != 0" | bc -l) )); then
                has_nonzero=true
                break
            fi
        done

        if [ "$has_nonzero" = true ]; then
            # Format first_val: keep original decimal format
            local formatted_val="$first_val"
            LAMBDA_SUFFIX="${LAMBDA_SUFFIX}_${suffix_name}${formatted_val}"
        fi
    }

    process_lambda "c2r" "lambda_c2r"
    process_lambda "wc2r" "lambda_wc2r"
    process_lambda "swc2r" "lambda_swc2r"
    process_lambda "sli" "lambda_sli"
    process_lambda "asg" "lambda_asg"

    # Process swc2r_alpha parameter
    SWC2R_ALPHA_VAL=$(echo "$CMD" | grep -oP "(?<=--swc2r_alpha )[^ ]+" || true)
    if [ -n "$SWC2R_ALPHA_VAL" ]; then
        # Convert integer to float format (e.g., 2 -> 2.0, 3 -> 3.0)
        if [[ "$SWC2R_ALPHA_VAL" =~ ^[0-9]+$ ]]; then
            SWC2R_ALPHA_FORMATTED="${SWC2R_ALPHA_VAL}.0"
        else
            SWC2R_ALPHA_FORMATTED="$SWC2R_ALPHA_VAL"
        fi
        LAMBDA_SUFFIX="${LAMBDA_SUFFIX}_alpha${SWC2R_ALPHA_FORMATTED}"
    fi

    # Process swc2r_tau parameter
    SWC2R_TAU_VAL=$(echo "$CMD" | grep -oP "(?<=--swc2r_tau )[^ ]+" || true)
    if [ -n "$SWC2R_TAU_VAL" ]; then
        LAMBDA_SUFFIX="${LAMBDA_SUFFIX}_tau${SWC2R_TAU_VAL}"
    fi

    # Process swc2r_type parameter
    SWC2R_TYPE_VAL=$(echo "$CMD" | grep -oP "(?<=--swc2r_type )[^ ]+" || true)
    if [ -n "$SWC2R_TYPE_VAL" ]; then
        LAMBDA_SUFFIX="${LAMBDA_SUFFIX}_${SWC2R_TYPE_VAL}"
    fi

    DTYPE_SUFFIX="_${DTYPE}"

    ACTUAL_SAVE_DIR="${RAW_SAVE_DIR}_${MODEL_NAME}_${ARCHS_JOINED}${LAMBDA_SUFFIX}${DTYPE_SUFFIX}"
    ACTUAL_SAVE_DIR=$(echo "$ACTUAL_SAVE_DIR" | tr '/' '_')

    echo "Parsed Info:"
    echo "  Actual Save Dir: $ACTUAL_SAVE_DIR"
    echo "  Layers:          $LAYERS"
    echo "  Device:          $DEVICE"
    echo "  Target GPU Memory: ${TARGET_GB}GB"

    # --- Check GPU Memory Before Training ---
    if [ "$TARGET_GB" -gt 0 ]; then
        GPU_ID=$(echo "$DEVICE" | sed 's/cuda://')
        echo "Checking GPU $GPU_ID memory availability..."

        # Get total GPU memory
        total_memory=140  # 140GB total memory per GPU

        # Wait loop for GPU memory
        while true; do
            current_usage=$(get_gpu_memory_usage "$GPU_ID")
            available_memory=$(echo "$total_memory - $current_usage" | bc -l)

            if (( $(echo "$available_memory >= $TARGET_GB" | bc -l) )); then
                echo "GPU $GPU_ID has sufficient memory: ${available_memory}GB available >= ${TARGET_GB}GB required"
                break
            else
                # Use \r to return to beginning of line and overwrite
                printf "\rGPU $GPU_ID insufficient memory: ${available_memory}GB available < ${TARGET_GB}GB required, waiting..."
                sleep 1
            fi
        done
        # Print newline after loop completes
        echo
        echo "GPU memory check passed, proceeding with training..."
    fi

    # --- PIPELINE STAGE 1: Training with detailed timing ---
    echo "[PIPELINE STAGE 1/2] =========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> Starting Training (with $TRAIN_ENV)..."
    echo "[PIPELINE] Job $((i+1)) Training Start"
    TRAIN_START_TIME=$(date +%s)
    (
        source "$TRAIN_ENV"
        cd "$TRAIN_DIR"
        echo "[TRAINING] Environment loaded, directory changed to $TRAIN_DIR"
        echo "[TRAINING] About to execute: $CMD --decay_start_fraction $DECAY_START_FRACTION"
        eval "$CMD --decay_start_fraction $DECAY_START_FRACTION"
    )

    if [ $? -ne 0 ]; then
        echo "[ERROR] Training failed! Exiting pipeline."
        exit 1
    fi

    TRAIN_END_TIME=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))
    echo "[PIPELINE] Job $((i+1)) Training Completed in ${TRAIN_DURATION} seconds"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> Training Completed."

    # --- PIPELINE STAGE 2: Evaluation with detailed timing ---
    echo "[PIPELINE STAGE 2/2] =========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> Starting Evaluation (with $EVAL_ENV)..."
    echo "[PIPELINE] Job $((i+1)) Evaluation Start"
    EVAL_START_TIME=$(date +%s)

    # Construct SAE Path for Eval using the reconstructed ACTUAL_SAVE_DIR
    SAE_PATH="../dictionary_learning_demo/$ACTUAL_SAVE_DIR/resid_post_layer_$LAYERS/trainer_{}/ae.pt"
    EVAL_DATASET_NAME=$(detect_dataset "$ACTUAL_SAVE_DIR")
    EVAL_DATASET_ARG=$(get_dataset_arg "$EVAL_DATASET_NAME")
    GPU_ID=$(echo "$DEVICE" | sed 's/cuda://')

    echo "[EVAL] Dataset: $EVAL_DATASET_NAME, SAE Path: $SAE_PATH"
    echo "[EVAL] GPU ID: $GPU_ID, Device: $DEVICE"

    (
        echo "[EVAL] Loading evaluation environment..."
        source "$EVAL_ENV"
        echo "[EVAL] Environment loaded, changing to $EVAL_DIR"
        cd "$EVAL_DIR"

        # Wait for GPU memory if needed
        if [ "$TARGET_GB" -gt 0 ]; then
             echo "[EVAL] Checking GPU $GPU_ID memory..."
             # Note: wait_for_gpu_memory is defined in parent, but subshell can access it
             # However, to be robust, we just print status
        fi

        echo "[EVAL] Starting Python evaluation script..."
        PYTHON_SCRIPT_NAME="sae_bench/custom_saes/run_eval_cli.py"
        python "$PYTHON_SCRIPT_NAME" \
            --device "$DEVICE" \
            --sae_dirs "$SAE_PATH" \
            --eval_types "${EVAL_TYPES[@]}" \
            --dir_len $DIR_LEN \
            --openai_api_key "$OPENAI_API_KEY" \
            --openai_base_url "$OPENAI_BASE_URL" \
            --openai_model "$OPENAI_MODEL" \
            --openai_max_workers $OPENAI_MAX_WORKERS \
            --target_gb $TARGET_GB \
            --random_seed $42 \
            --sae_dtype "$DTYPE" \
            $REVERSE_FLAG \
            $EVAL_DATASET_ARG
    )

    if [ $? -ne 0 ]; then
        echo "[ERROR] Evaluation failed!"
        exit 1
    fi

    EVAL_END_TIME=$(date +%s)
    EVAL_DURATION=$((EVAL_END_TIME - EVAL_START_TIME))
    TOTAL_DURATION=$((EVAL_END_TIME - TRAIN_START_TIME))

    echo "[PIPELINE] Job $((i+1)) Evaluation Completed in ${EVAL_DURATION} seconds"
    echo "[PIPELINE] Job $((i+1)) Total Time: ${TOTAL_DURATION} seconds (Training: ${TRAIN_DURATION}s, Evaluation: ${EVAL_DURATION}s)"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> Evaluation Completed."

    ) &  # Close subshell and run in background
done

# Wait for all background jobs to complete
wait

# Final summary
echo "[PIPELINE] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All pipeline jobs finished!"
echo "[PIPELINE] Summary: All jobs completed successfully"

