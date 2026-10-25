"""End-to-end entry point: python -m src.run_all"""

from . import data, step0
from .common import DATA


def main():
    if not (DATA / "seqs_1024.npz").exists():
        data.build(n_per_domain=100, seq_len=1024)
    step0.run()


if __name__ == "__main__":
    main()
