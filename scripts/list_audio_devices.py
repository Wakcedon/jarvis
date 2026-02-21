from __future__ import annotations

import json

import sounddevice as sd


def main() -> None:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    default_in, default_out = sd.default.device

    out = []
    for i, d in enumerate(devices):
        ha = hostapis[d["hostapi"]]["name"] if "hostapi" in d else "?"
        out.append(
            {
                "index": i,
                "name": d.get("name"),
                "hostapi": ha,
                "max_input_channels": d.get("max_input_channels"),
                "max_output_channels": d.get("max_output_channels"),
                "default_samplerate": d.get("default_samplerate"),
                "is_default_input": i == default_in,
                "is_default_output": i == default_out,
            }
        )

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
