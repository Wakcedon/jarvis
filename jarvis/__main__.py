from __future__ import annotations

import argparse

from jarvis.app import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный голосовой ассистент (RU)")
    parser.add_argument("--config", default="config.yaml", help="Путь к config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
