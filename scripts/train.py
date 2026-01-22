import argparse

from recsys.runner import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a recsys model from a config.")
    parser.add_argument(
        "--config",
        "-c",
        default="conf/deepfm.yaml",
        help="Path to the config YAML file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed from the config file.",
    )
    args = parser.parse_args()
    train_from_config(args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
