from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class AudioModel(BaseModel):
    sample_rate: PositiveInt = 16000
    input_device: int | str | None = None
    output_device: int | str | None = None
    activation_feedback: Literal["tts", "beep", "none"] = "tts"
    activation_text: str = "Слушаю."
    activation_delay_s: float = Field(default=0.05, ge=0.0, le=2.0)
    barge_in_enabled: bool = True
    barge_in_rms_threshold: float = Field(default=0.02, ge=0.0, le=1.0)


class WakeWordModel(BaseModel):
    enabled: bool = True
    phrases: list[str] = Field(default_factory=lambda: ["джарвис", "ассистент"])
    vosk_model_path: str = "models/vosk-model-small-ru-0.22"
    blocksize: PositiveInt = 1600
    min_rms: float = Field(default=0.0, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    use_partial: bool = True
    partial_hits: int = Field(default=2, ge=1, le=10)
    cooldown_s: float = Field(default=1.2, ge=0.0, le=10.0)
    confirm_window_s: float = Field(default=1.0, ge=0.1, le=5.0)
    noise_gate_multiplier: float = Field(default=3.2, ge=1.0, le=20.0)
    noise_ema_alpha: float = Field(default=0.02, ge=0.001, le=0.2)


class RecordingModel(BaseModel):
    max_s: PositiveFloat = 8.0
    start_threshold: float = Field(default=0.015, ge=0.0, le=1.0)
    auto_threshold: bool = False
    noise_calibrate_s: float = Field(default=0.4, ge=0.0, le=10.0)
    threshold_multiplier: float = Field(default=3.5, ge=1.0, le=20.0)
    start_immediately_after_wakeword: bool = True
    pre_roll_s: float = Field(default=0.25, ge=0.0, le=2.0)
    auto_gain: bool = True
    target_rms: float = Field(default=0.03, ge=0.0, le=1.0)
    max_gain: float = Field(default=25.0, ge=1.0, le=200.0)
    end_silence_s: float = Field(default=0.6, ge=0.05, le=5.0)
    chunk_ms: int = Field(default=20, ge=5, le=200)


class STTModel(BaseModel):
    backend: Literal["faster-whisper"] = "faster-whisper"
    model_size: str = "base"
    model_path: str | None = None
    device: Literal["cpu", "cuda"] = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = Field(default=4, ge=1, le=256)
    num_workers: int = Field(default=1, ge=1, le=32)
    language: str = "ru"
    beam_size: int = Field(default=3, ge=1, le=10)
    vad_filter: bool = True
    initial_prompt: str = "Разговор на русском языке."
    min_avg_logprob: float = Field(default=-1.2, ge=-10.0, le=0.0)
    max_no_speech_prob: float = Field(default=0.6, ge=0.0, le=1.0)


class TTSModel(BaseModel):
    backend: Literal["piper"] = "piper"
    model_path: str = "models/piper/ru_RU-dmitri-medium.onnx"
    config_path: str = "models/piper/ru_RU-dmitri-medium.onnx.json"
    speaker: str | None = None
    length_scale: float = Field(default=0.88, ge=0.5, le=2.0)
    sentence_silence: float = Field(default=0.15, ge=0.0, le=1.0)
    volume: float = Field(default=1.0, ge=0.0, le=2.0)


class LLMModel(BaseModel):
    enabled: bool = True
    backend: Literal["ollama", "llamacpp"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    dual_model_enabled: bool = False
    fast_model: str = "qwen2.5:0.5b-instruct"
    quality_model: str = "qwen2.5:1.5b-instruct"
    timeout_s: int = Field(default=60, ge=1, le=600)
    disable_if_unreachable: bool = True
    autostart: bool = True
    auto_pull_model: bool = False
    num_predict: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    history_max_messages: int = Field(default=18, ge=1, le=200)
    llamacpp_base_url: str = "http://localhost:8080"
    ollama_num_ctx: int = Field(default=1024, ge=128, le=32768)
    ollama_num_thread: int = Field(default=4, ge=1, le=256)
    ollama_keep_alive: str = "30m"
    fast_mode: bool = True
    stream_to_tts: bool = True
    fast_max_sentences: int = Field(default=1, ge=1, le=8)
    fast_max_chars: int = Field(default=260, ge=64, le=4000)
    warmup_on_start: bool = True


class CommandsModel(BaseModel):
    notes_path: str = "data/notes.md"
    memory_path: str = "data/memory.json"
    weather_cache_path: str = "data/weather_cache.json"
    reminders_path: str = "data/reminders.json"
    todo_path: str = "data/todo.json"
    screenshots_dir: str = "data/screenshots"
    default_browser_url: str = "https://www.google.com"
    default_city: str = "Новокузнецк"
    allowed_apps: list[str] = Field(default_factory=lambda: ["firefox", "google-chrome", "chromium", "code"])
    allow_shell: bool = False
    allow_desktop_launch: bool = True


class UIModel(BaseModel):
    mic_level_threshold: float = Field(default=0.02, ge=0.0, le=1.0)
    # "auto" uses the XDG state dir (e.g. ~/.local/state/jarvis/status.json).
    status_path: str = "auto"


class StorageModel(BaseModel):
    backend: Literal["sqlite", "files"] = "sqlite"
    db_path: str | None = None


class CapabilitiesModel(BaseModel):
    can_open_urls: bool = True
    can_open_paths: bool = True
    can_launch_apps: bool = True
    can_desktop_launch: bool = True
    can_volume: bool = True
    can_mic_mute: bool = True
    can_network: bool = True
    can_media: bool = True
    can_window: bool = True
    can_clipboard: bool = True
    can_screenshots: bool = True
    can_shell: bool = False
    can_power: bool = True


class ConfigModel(BaseModel):
    audio: AudioModel = Field(default_factory=AudioModel)
    wake_word: WakeWordModel = Field(default_factory=WakeWordModel)
    recording: RecordingModel = Field(default_factory=RecordingModel)
    stt: STTModel = Field(default_factory=STTModel)
    tts: TTSModel = Field(default_factory=TTSModel)
    llm: LLMModel = Field(default_factory=LLMModel)
    commands: CommandsModel = Field(default_factory=CommandsModel)
    ui: UIModel = Field(default_factory=UIModel)
    storage: StorageModel = Field(default_factory=StorageModel)
    capabilities: CapabilitiesModel = Field(default_factory=CapabilitiesModel)
