import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/poc_em_cl.yaml")
    parser.add_argument("--dataset", type=str, default="chexpert")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Running fine-tuning baseline")
    print("Dataset:", args.dataset)
    print("Config project:", config["project"]["name"])
    print("TODO: implement training loop")


if __name__ == "__main__":
    main()
