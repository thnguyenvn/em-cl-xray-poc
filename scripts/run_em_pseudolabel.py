import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/poc_em_cl.yaml")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    print("Running EM pseudo-labeling")
    print("Pseudo-label threshold:", args.threshold)
    print("TODO: implement EM pseudo-labeling pipeline")


if __name__ == "__main__":
    main()
