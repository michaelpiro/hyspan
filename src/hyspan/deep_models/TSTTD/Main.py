import yaml
from .Train_eval import train, eval_model, select_best
import os


config_dictionary = {
    # Execution State
    "state": "train",  # train, eval, or select_best
    "device": "cuda:0",  # Change to "cpu" if you don't have a GPU
    "seed": 1,

    "data_keys":
    # Dataset Configuration
        {
            "data": "data",  # Change if your .mat uses 'img', 'cube', etc.
            "map": "map",  # Change if your .mat uses 'gt', 'label', etc.
        },

    "model":
    # Model Architecture
        {
            "band": 189,
            "group_length": 20,
            "depth": 4,
            "heads": 4,
            "dim_head": 64,
            "mlp_dim": 64,
            "channel": 128,
            "adjust": False,
        },

    "training":
    # Hyperparameters
        {
            "epochs": 20,
            "batch_size": 64,
            "lr": 1.0e-4,
            "multiplier": 2,
            "epision": 5,
            "grad_clip": 1.0,
            "save_freq": 1,  # Save model every m epochs
        },

    "paths":
    # Paths and Checkpoints
        {
            "save_dir": "./Checkpoint/",
            "run_name": "experiment_1",  # Added to clean up checkpoint saving
            "dataset_path": "",
            "training_load_weight": None,
            "test_load_weight": "ckpt_5_.pt",
        },
}


def main(config_path="train_config.yaml", config_dict=None):
    if config_dict is not None:
        config = config_dict
    else:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

    state = config.get("state", "eval")
    print(f"--- Running in {state.upper()} mode ---")

    if state == "train":
        train(config)
    elif state == "eval":
        eval_model(config)
    else:
        select_best(config)


if __name__ == '__main__':
    main()
