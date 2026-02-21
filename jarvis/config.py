from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    input_device: int | str | None = None
    output_device: int | str | None = None
    activation_feedback: str = "tts"  # tts|beep|none
    activation_text: str = "Слушаю."
    activation_delay_s: float = 0.05


@dataclass(frozen=True)
class WakeWordConfig:
    enabled: bool = True
    phrases: tuple[str, ...] = ("джарвис", "ассистент")
    vosk_model_path: Path = PROJECT_ROOT / "models" / "vosk-model-small-ru-0.22"
    blocksize: int = 1600
    min_rms: float = 0.0


@dataclass(frozen=True)
class RecordingConfig:
    max_s: float = 8.0
    start_threshold: float = 0.015
    auto_threshold: bool = False
    noise_calibrate_s: float = 0.4
    threshold_multiplier: float = 3.5
    start_immediately_after_wakeword: bool = True
    pre_roll_s: float = 0.25
    auto_gain: bool = True
    target_rms: float = 0.03
    max_gain: float = 25.0
    end_silence_s: float = 0.6
    chunk_ms: int = 20


@dataclass(frozen=True)
class STTConfig:
    backend: str = "faster-whisper"
    model_size: str = "base"
    model_path: Path | None = None
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 4
    num_workers: int = 1
    language: str = "ru"
    beam_size: int = 3
    vad_filter: bool = True
    initial_prompt: str = "Разговор на русском языке."
    min_avg_logprob: float = -1.2
    max_no_speech_prob: float = 0.6


@dataclass(frozen=True)
class TTSConfig:
    backend: str = "piper"
    model_path: Path = PROJECT_ROOT / "models" / "piper" / "ru_RU-dmitri-medium.onnx"
    config_path: Path = PROJECT_ROOT / "models" / "piper" / "ru_RU-dmitri-medium.onnx.json"
    speaker: str | None = None
    length_scale: float = 0.88
    sentence_silence: float = 0.15
    volume: float = 1.0


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = True
    backend: str = "ollama"  # ollama|llamacpp
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    dual_model_enabled: bool = False
    fast_model: str = "qwen2.5:0.5b-instruct"
    quality_model: str = "qwen2.5:1.5b-instruct"
    timeout_s: int = 60
    disable_if_unreachable: bool = True
    autostart: bool = True
    auto_pull_model: bool = False
    num_predict: int = 256
    temperature: float = 0.2
    history_max_messages: int = 18
    llamacpp_base_url: str = "http://localhost:8080"
    ollama_num_ctx: int = 1024
    ollama_num_thread: int = 4
    ollama_keep_alive: str = "30m"
    fast_mode: bool = True
    fast_max_sentences: int = 1
    fast_max_chars: int = 260
    warmup_on_start: bool = True


@dataclass(frozen=True)
class CommandsConfig:
    notes_path: Path = PROJECT_ROOT / "data" / "notes.md"
    memory_path: Path = PROJECT_ROOT / "data" / "memory.json"
    weather_cache_path: Path = PROJECT_ROOT / "data" / "weather_cache.json"
    reminders_path: Path = PROJECT_ROOT / "data" / "reminders.json"
    todo_path: Path = PROJECT_ROOT / "data" / "todo.json"
    screenshots_dir: Path = PROJECT_ROOT / "data" / "screenshots"
    default_browser_url: str = "https://www.google.com"
    default_city: str = "Новокузнецк"
    allowed_apps: tuple[str, ...] = ("firefox", "google-chrome", "chromium", "code")
    allow_shell: bool = False
    allow_desktop_launch: bool = True


@dataclass(frozen=True)
class UIConfig:
    mic_level_threshold: float = 0.02
    status_path: Path = PROJECT_ROOT / "runtime" / "status.json"


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = AudioConfig()
    wake_word: WakeWordConfig = WakeWordConfig()
    recording: RecordingConfig = RecordingConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()
    llm: LLMConfig = LLMConfig()
    commands: CommandsConfig = CommandsConfig()
    ui: UIConfig = UIConfig()


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def load_config(config_path: str | Path) -> Config:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    audio_raw = _get(raw, "audio", {})
    wake_raw = _get(raw, "wake_word", {})
    rec_raw = _get(raw, "recording", {})
    stt_raw = _get(raw, "stt", {})
    tts_raw = _get(raw, "tts", {})
    llm_raw = _get(raw, "llm", {})
    cmd_raw = _get(raw, "commands", {})
    ui_raw = _get(raw, "ui", {})

    audio = AudioConfig(
        sample_rate=int(_get(audio_raw, "sample_rate", 16000)),
        input_device=_get(audio_raw, "input_device", None),
        output_device=_get(audio_raw, "output_device", None),
        activation_feedback=str(_get(audio_raw, "activation_feedback", "tts")),
        activation_text=str(_get(audio_raw, "activation_text", "Да, слушаю.")),
        activation_delay_s=float(_get(audio_raw, "activation_delay_s", 0.15)),
    )

    wake = WakeWordConfig(
        enabled=bool(_get(wake_raw, "enabled", True)),
        phrases=tuple(str(x).lower() for x in _get(wake_raw, "phrases", ["джарвис", "ассистент"])),
        vosk_model_path=_resolve_path(_get(wake_raw, "vosk_model_path", "models/vosk-model-small-ru-0.22")),
        blocksize=int(_get(wake_raw, "blocksize", 1600)),
        min_rms=float(_get(wake_raw, "min_rms", 0.0)),
    )

    recording = RecordingConfig(
        max_s=float(_get(rec_raw, "max_s", 8.0)),
        start_threshold=float(_get(rec_raw, "start_threshold", 0.015)),
        auto_threshold=bool(_get(rec_raw, "auto_threshold", False)),
        noise_calibrate_s=float(_get(rec_raw, "noise_calibrate_s", 0.4)),
        threshold_multiplier=float(_get(rec_raw, "threshold_multiplier", 3.5)),
        start_immediately_after_wakeword=bool(_get(rec_raw, "start_immediately_after_wakeword", True)),
        pre_roll_s=float(_get(rec_raw, "pre_roll_s", 0.25)),
        auto_gain=bool(_get(rec_raw, "auto_gain", True)),
        target_rms=float(_get(rec_raw, "target_rms", 0.03)),
        max_gain=float(_get(rec_raw, "max_gain", 25.0)),
        end_silence_s=float(_get(rec_raw, "end_silence_s", 0.6)),
        chunk_ms=int(_get(rec_raw, "chunk_ms", 20)),
    )

    stt = STTConfig(
        backend=str(_get(stt_raw, "backend", "faster-whisper")),
        model_size=str(_get(stt_raw, "model_size", "base")),
        model_path=_resolve_optional_path(_get(stt_raw, "model_path", None)),
        device=str(_get(stt_raw, "device", "cpu")),
        compute_type=str(_get(stt_raw, "compute_type", "int8")),
        cpu_threads=int(_get(stt_raw, "cpu_threads", 4)),
        num_workers=int(_get(stt_raw, "num_workers", 1)),
        language=str(_get(stt_raw, "language", "ru")),
        beam_size=int(_get(stt_raw, "beam_size", 3)),
        vad_filter=bool(_get(stt_raw, "vad_filter", True)),
        initial_prompt=str(_get(stt_raw, "initial_prompt", "Разговор на русском языке.")),
        min_avg_logprob=float(_get(stt_raw, "min_avg_logprob", -1.2)),
        max_no_speech_prob=float(_get(stt_raw, "max_no_speech_prob", 0.6)),
    )

    tts = TTSConfig(
        backend=str(_get(tts_raw, "backend", "piper")),
        model_path=_resolve_path(_get(tts_raw, "model_path", "models/piper/ru_RU-dmitri-medium.onnx")),
        config_path=_resolve_path(_get(tts_raw, "config_path", "models/piper/ru_RU-dmitri-medium.onnx.json")),
        speaker=_get(tts_raw, "speaker", None),
        length_scale=float(_get(tts_raw, "length_scale", 0.88)),
        sentence_silence=float(_get(tts_raw, "sentence_silence", 0.15)),
        volume=float(_get(tts_raw, "volume", 1.0)),
    )

    llm = LLMConfig(
        enabled=bool(_get(llm_raw, "enabled", True)),
        backend=str(_get(llm_raw, "backend", "ollama")),
        base_url=str(_get(llm_raw, "base_url", "http://localhost:11434")),
        model=str(_get(llm_raw, "model", "mistral")),
        dual_model_enabled=bool(_get(llm_raw, "dual_model_enabled", False)),
        fast_model=str(_get(llm_raw, "fast_model", "qwen2.5:0.5b-instruct")),
        quality_model=str(_get(llm_raw, "quality_model", "qwen2.5:1.5b-instruct")),
        timeout_s=int(_get(llm_raw, "timeout_s", 60)),
        disable_if_unreachable=bool(_get(llm_raw, "disable_if_unreachable", True)),
        autostart=bool(_get(llm_raw, "autostart", True)),
        auto_pull_model=bool(_get(llm_raw, "auto_pull_model", False)),
        num_predict=int(_get(llm_raw, "num_predict", 256)),
        temperature=float(_get(llm_raw, "temperature", 0.2)),
        history_max_messages=int(_get(llm_raw, "history_max_messages", 18)),
        llamacpp_base_url=str(_get(llm_raw, "llamacpp_base_url", "http://localhost:8080")),
        ollama_num_ctx=int(_get(llm_raw, "ollama_num_ctx", 1024)),
        ollama_num_thread=int(_get(llm_raw, "ollama_num_thread", 4)),
        ollama_keep_alive=str(_get(llm_raw, "ollama_keep_alive", "30m")),
        fast_mode=bool(_get(llm_raw, "fast_mode", True)),
        fast_max_sentences=int(_get(llm_raw, "fast_max_sentences", 1)),
        fast_max_chars=int(_get(llm_raw, "fast_max_chars", 260)),
        warmup_on_start=bool(_get(llm_raw, "warmup_on_start", True)),
    )

    commands = CommandsConfig(
        notes_path=_resolve_path(_get(cmd_raw, "notes_path", "data/notes.md")),
        memory_path=_resolve_path(_get(cmd_raw, "memory_path", "data/memory.json")),
        weather_cache_path=_resolve_path(_get(cmd_raw, "weather_cache_path", "data/weather_cache.json")),
        reminders_path=_resolve_path(_get(cmd_raw, "reminders_path", "data/reminders.json")),
        todo_path=_resolve_path(_get(cmd_raw, "todo_path", "data/todo.json")),
        screenshots_dir=_resolve_path(_get(cmd_raw, "screenshots_dir", "data/screenshots")),
        default_browser_url=str(_get(cmd_raw, "default_browser_url", "https://www.google.com")),
        default_city=str(_get(cmd_raw, "default_city", "Новокузнецк")),
        allowed_apps=tuple(str(x) for x in _get(cmd_raw, "allowed_apps", ["firefox", "google-chrome", "chromium", "code"])),
        allow_shell=bool(_get(cmd_raw, "allow_shell", False)),
        allow_desktop_launch=bool(_get(cmd_raw, "allow_desktop_launch", True)),
    )

    ui = UIConfig(
        mic_level_threshold=float(_get(ui_raw, "mic_level_threshold", 0.02)),
        status_path=_resolve_path(_get(ui_raw, "status_path", "runtime/status.json")),
    )

    return Config(audio=audio, wake_word=wake, recording=recording, stt=stt, tts=tts, llm=llm, commands=commands, ui=ui)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return _resolve_path(s)
