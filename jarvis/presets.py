from __future__ import annotations

from pathlib import Path

from jarvis.yaml_config import set_dotted
from jarvis.paths import ensure_default_config


def apply_speed_preset(config_path: Path, *, template_path: Path) -> None:
    ensure_default_config(template_path=template_path, dest_path=config_path)

    set_dotted(config_path, "wake_word.use_partial", "true")
    set_dotted(config_path, "wake_word.partial_hits", "1")
    set_dotted(config_path, "wake_word.cooldown_s", "0.8")

    set_dotted(config_path, "recording.max_s", "4.0")
    set_dotted(config_path, "recording.end_silence_s", "0.35")
    set_dotted(config_path, "audio.activation_delay_s", "0.10")

    set_dotted(config_path, "stt.beam_size", "1")

    set_dotted(config_path, "llm.num_predict", "48")
    set_dotted(config_path, "llm.ollama_num_ctx", "768")
    set_dotted(config_path, "llm.stream_to_tts", "true")

    set_dotted(config_path, "tts.length_scale", "0.84")
    set_dotted(config_path, "tts.sentence_silence", "0.12")


def apply_room_preset(config_path: Path, *, template_path: Path) -> None:
    ensure_default_config(template_path=template_path, dest_path=config_path)

    set_dotted(config_path, "wake_word.use_partial", "true")
    set_dotted(config_path, "wake_word.partial_hits", "2")
    set_dotted(config_path, "wake_word.min_confidence", "0.65")
    set_dotted(config_path, "wake_word.noise_gate_multiplier", "3.8")
    set_dotted(config_path, "wake_word.min_rms", "0.004")

    set_dotted(config_path, "recording.auto_gain", "true")
    set_dotted(config_path, "recording.target_rms", "0.05")
    set_dotted(config_path, "recording.max_gain", "60")
    set_dotted(config_path, "recording.pre_roll_s", "0.35")

    set_dotted(config_path, "stt.beam_size", "3")
    set_dotted(config_path, "stt.vad_filter", "false")

    set_dotted(config_path, "audio.activation_feedback", "beep")
    set_dotted(config_path, "audio.activation_delay_s", "0.15")
