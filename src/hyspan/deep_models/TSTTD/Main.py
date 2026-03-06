import yaml
from .Train_eval import train, eval_model, select_best
import os
current_dir = os.path.dirname(__file__)
config_file_path = os.path.join(current_dir, "config.yaml")
def main(config_path=config_file_path):
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