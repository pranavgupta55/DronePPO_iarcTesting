"""python3 -m drone_rl <command>: train-one, batch, eval, imitate, baseline, check-scheme, index, report, bench, view."""
import sys

from drone_rl.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
