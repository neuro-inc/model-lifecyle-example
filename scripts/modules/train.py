import argparse
from char_rnn_classification_tutorial import train_eval_save


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample training script")
    parser.add_argument("-d", "--data_dir", required=True, help="Path to the dataset")
    parser.add_argument("-m", "--model_dir", required=True, help="Path to save the model")
    return parser


def train(args: argparse.Namespace) -> None:
    train_eval_save(args.data_dir, args.model_dir)


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
