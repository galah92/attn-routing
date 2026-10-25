"""End-to-end entry point: python -m src.run_all"""

from . import data, step0, step1
from .common import DATA


def main():
    built = False
    if not (DATA / "seqs_1024.npz").exists():
        data.build(n_per_domain=100, seq_len=1024)
        built = True
    _, domains = data.load(1024)
    data.make_split(domains)
    step0.run()
    step1.run()
    if built:
        data.hard_exit()  # streaming threads abort normal interpreter teardown


if __name__ == "__main__":
    main()
