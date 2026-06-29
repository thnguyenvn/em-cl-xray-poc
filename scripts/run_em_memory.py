import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/poc_em_cl.yaml")
    parser.add_argument("--memory_size", type=int, default=1000)
    parser.add_argument("--em_components", type=int, default=5)
    args = parser.parse_args()

    print("Running EM-guided replay memory")
    print("Memory size:", args.memory_size)
    print("EM components:", args.em_components)
    print("TODO: implement EM-guided memory selection")


if __name__ == "__main__":
    main()
