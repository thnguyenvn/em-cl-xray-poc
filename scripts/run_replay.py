import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/poc_em_cl.yaml")
    parser.add_argument("--memory_size", type=int, default=1000)
    args = parser.parse_args()

    print("Running random replay baseline")
    print("Memory size:", args.memory_size)
    print("TODO: implement replay training loop")


if __name__ == "__main__":
    main()
