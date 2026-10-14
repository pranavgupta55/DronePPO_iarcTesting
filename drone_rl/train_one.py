"""Runs one training session: python3 -m drone_rl train-one <spec.json> <run_dir>"""
import json
import sys


def main():
    from drone_rl.trainer import train  # imported here so the spawned env workers (which re-import this file) don't load torch
    spec = json.load(open(sys.argv[1]))
    done = train(spec, sys.argv[2])
    print("done", json.dumps({k: v for k, v in done.items() if k != "last"}), flush=True)


if __name__ == "__main__":
    main()
