from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_BY_SIZE = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Скачать faster-whisper модель в локальную папку")
    p.add_argument("--size", default="base", choices=sorted(REPO_BY_SIZE.keys()))
    p.add_argument("--out", default="models/whisper/faster-whisper-base")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    repo = REPO_BY_SIZE[args.size]
    print(f"Скачиваю {repo} → {out}")

    snapshot_download(
        repo_id=repo,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        # безопасно оставить allow_patterns пустым — скачает нужные файлы модели
    )

    print("OK")


if __name__ == "__main__":
    main()
