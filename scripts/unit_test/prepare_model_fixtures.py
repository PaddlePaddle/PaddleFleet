# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
"""Give each test container private checkpoint metadata beside shared weights."""

import argparse
import os
import shutil
from pathlib import Path


def prepare_model_fixtures(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError(
            "Fixture destination must be outside the shared source"
        )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Fixture destination must be empty")

    def link_file(src: str, dst: str) -> str:
        os.symlink(os.path.realpath(src), dst)
        return dst

    shutil.copytree(
        source,
        destination,
        copy_function=link_file,
        ignore=shutil.ignore_patterns("*.auto_generated.metadata"),
        dirs_exist_ok=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare_model_fixtures(args.source, args.destination)
