from argparse import ArgumentParser
import config


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ucorecsys",
        description="Tool to train, evaluate, or perform inference with a recommendation model.",
    )

    # 1) Mutually exclusive group for execution mode: inference, train, eval, surprise
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-i",
        "--inference",
        metavar="MODEL_PATH",
        help=(
            "Run inference on an already-trained model. "
            "You must specify the path to the model file."
        ),
    )
    mode_group.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train the model using the specified parameters.",
    )
    mode_group.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Evaluate the proposed model (using the dataset and cvtype parameters).",
    )
    mode_group.add_argument(
        "-s",
        "--surprise",
        action="store_true",
        help="Run evaluation of Surprise algorithms (using the Surprise library).",
    )

    # 2) Group for data / validation options
    data_group = parser.add_argument_group("Data / Validation Options")
    data_group.add_argument(
        "-ds",
        "--dataset",
        choices=config.DATASETS_CHOICES,
        default=config.DATASET,
        help=f"Name of the dataset to use. Default: '{config.DATASET}'.",
    )
    data_group.add_argument(
        "-cv",
        "--cvtype",
        choices=config.CV_TYPES_CHOICES,
        default=config.CV_TYPE,
        help=f"Type of cross-validation. Default: '{config.CV_TYPE}'.",
    )

    # 3) Group for training options
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"Number of epochs for training (default: {config.EPOCHS}).",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Batch size for training (default: {config.BATCH_SIZE}).",
    )
    train_group.add_argument(
        "--output-model",
        metavar="MODEL_OUT",
        default=config.OUTPUT_MODEL_PATH,
        help=f"Path to save the trained model (default: {config.OUTPUT_MODEL_PATH}).",
    )
    train_group.add_argument(
        "--k_ranking",
        type=int,
        default=config.K,
        help=f"Parameter k for ranking metrics calculation (default: {config.K}).",
    )

    # 4) Group for evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument(
        "--balance",
        action="store_true",
        default=config.BALANCE,
        help="Balance training and validation data.",
    )
    eval_group.add_argument(
        "--k",
        type=int,
        default=config.K,
        help=f"Parameter k for ranking metrics calculation (default: {config.K}).",
    )
    eval_group.add_argument(
        "-ns",
        "--splits",
        type=int,
        default=config.K_FOLD,
        help=f"Number of splits for cross-validation (default: {config.K_FOLD}).",
    )

    # 5) Group for general / miscellaneous options
    misc_group = parser.add_argument_group("General Options")
    misc_group.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Generate metric plots during execution.",
        default=config.PLOT,
    )
    misc_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed information during execution.",
    )

    return parser
