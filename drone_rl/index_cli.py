"""python3 -m drone_rl index: rebuild runs/index.json."""
from drone_rl import registry


def main():
    rows = registry.build_index()
    print(f"indexed {len(rows)} runs -> {registry.RUNS / 'index.json'}")


if __name__ == "__main__":
    main()
