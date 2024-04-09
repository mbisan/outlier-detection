from argparse import ArgumentParser
import multiprocessing
from dataclasses import dataclass
from typing import Tuple

# pylint: disable=line-too-long invalid-name

@dataclass
class Arguments:
    dataset: str = ""
    dataset_dir: str = "./datasets"
    batch_size: int = 32
    num_workers: int = multiprocessing.cpu_count() // 2

def get_parser():
    parser = ArgumentParser()

    for argument, typehint in Arguments.__annotations__.items():
        if typehint == bool:
            parser.add_argument(
                f"--{argument}",
                action="store_false" if Arguments.__dict__[argument] else "store_true")
        elif typehint == Tuple[int]:
            parser.add_argument(
                f"--{argument}", nargs="+", type=int, default=Arguments.__dict__[argument])
        else:
            parser.add_argument(
                f"--{argument}", type=typehint, default=Arguments.__dict__[argument])

    return parser
