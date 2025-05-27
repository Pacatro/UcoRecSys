from argparse import ArgumentParser

model_parser = ArgumentParser(prog="ucorecsys")

model_parser.add_argument(
    "-i", "--inference", action="store_true", help="Run inference"
)
model_parser.add_argument(
    "-e",
    "--eval",
    action="store",
    help="Evaluate the proposed model, if -s is activate, then performs the type of evaluation for surprise algorithms",
    choices=["kfold", "loo"],
)
model_parser.add_argument(
    "-s",
    "--surprise",
    action="store_true",
    help="Evaluate the surprise algorithms",
)
model_parser.add_argument(
    "-ds",
    "--dataset",
    action="store",
    help="Name of the dataset to load",
    choices=["mars", "coursera", "itm"],
    default="mars",
)
model_parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Prints additional information",
    default=True,
)
