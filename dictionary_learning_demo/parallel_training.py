import subprocess
import time
import os



MODEL_NAME = "EleutherAI/pythia-160m-deduped"
MODEL_NAME = "google/gemma-2-2b"

MODEL_NAME = None

if "gemma" in MODEL_NAME:
    layer = 12
elif "pythia-70m" in MODEL_NAME:
    layer = 3
elif "pythia-160m" in MODEL_NAME:
    layer = 8
else:
    raise ValueError("Unknown model name")


configurations = [
    {
        "arch": "jump_relu",
        "layers": layer,
        "device": "cuda:0",
        "save_checkpoints": False
    },
    {
        "arch": "top_k p_anneal",
        "layers": layer,
        "device": "cuda:1",
        "save_checkpoints": False
    },
    {
        "arch": "batch_top_k standard_new",
        "layers": layer,
        "device": "cuda:2",
        "save_checkpoints": False
    },
    {
        "arch": "gated",
        "layers": layer,
        "device": "cuda:3",
        "save_checkpoints": False
    },
]


SAVE_DIR = "trained_saes/"

os.makedirs("logs", exist_ok=True)

for i, config in enumerate(configurations):
    log_file = f"logs/{(config['arch'].replace(' ', '_'))}_l{config['layers']}_{config['device'].replace(':', '_')}.out"

    if config["save_checkpoints"]:
        save_command = "--save_checkpoints"
    else:
        save_command = ""

    cmd = [
        "python", "demo.py",
        "--save_dir", SAVE_DIR,
        "--model_name", MODEL_NAME,
        "--architectures", config["arch"],
        "--layers", str(config["layers"]),
        "--device", config["device"],
        save_command,
    ]

    print(" ".join(cmd))

    with open(log_file, "w") as f:
        subprocess.Popen(
            f"nohup {' '.join(cmd)} > {log_file} 2>&1",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    print(f"Started job {i + 1}/{len(configurations)}: {config['arch']} with layers: {config['layers']}")
    time.sleep(2)

print("All jobs submitted!")
