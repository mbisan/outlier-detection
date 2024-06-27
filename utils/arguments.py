from argparse import ArgumentParser
from typing import Tuple

# pylint: disable=line-too-long invalid-name

def get_parser(args_dataclass) -> ArgumentParser:
    parser = ArgumentParser()

    for argument, typehint in args_dataclass.__annotations__.items():
        if typehint == bool:
            parser.add_argument(
                f"--{argument}",
                action="store_false" if args_dataclass.__dict__[argument] else "store_true")
        elif typehint == Tuple[int]:
            parser.add_argument(
                f"--{argument}", nargs="+", type=int, default=args_dataclass.__dict__[argument])
        else:
            parser.add_argument(
                f"--{argument}", type=typehint, default=args_dataclass.__dict__[argument])

    return parser
