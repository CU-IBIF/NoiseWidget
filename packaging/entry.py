"""Entry point for the packaged executable.

A frozen build has no console to pass arguments through on macOS or Windows, so
double-clicking must land somewhere useful. It opens the widget directly, while
still honouring command-line arguments when someone runs the binary from a shell.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # PyInstaller one-file builds re-execute the binary for child processes; without
    # this a stray multiprocessing call would fork the whole app repeatedly.
    multiprocessing.freeze_support()

    from noisewidget.__main__ import main as cli

    argv = sys.argv[1:]
    if not argv:
        argv = ["--ui"]
    return cli(argv)


if __name__ == "__main__":
    sys.exit(main())
