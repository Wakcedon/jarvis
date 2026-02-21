from __future__ import annotations

import sys
import json
import time
import shutil
import subprocess
import re
import threading

from jarvis.audio import beep, record_utterance
from jarvis.commands import (
    CommandRouter,
    PendingSystemActionResponse,
    RepeatLastHeardRequest,
    SpeakStatusRequest,
    TimerCancelResponse,
    TimerSetResponse,
    TimerStatusRequest,
    VerbositySetResponse,
)
from jarvis.config import load_config
from jarvis.llm_ollama import ChatMessage, OllamaClient, OllamaUnavailable
from jarvis.llm_llamacpp import ChatMessage as LlamaMessage, LlamaCppClient, LlamaCppUnavailable
from jarvis.stt_faster_whisper import FasterWhisperSTT
from jarvis.status import StatusWriter
from jarvis.tts_piper import PiperTTS, PiperVoice
from jarvis.wakeword_vosk import VoskWakeWordDetector


SYSTEM_PROMPT = (
    "Ты — локальный голосовой ассистент. Отвечай по-русски. "
    "По умолчанию — коротко (1–2 предложения). Если пользователь просит подробнее — 3–6 предложений."
)


def run(config_path: str) -> None:
    cfg = load_config(config_path)

    status = StatusWriter(path=cfg.ui.status_path)
    status.update(state="starting", message="Запуск…", force=True)

    router = CommandRouter(
        notes_path=cfg.commands.notes_path,
        memory_path=cfg.commands.memory_path,
        weather_cache_path=cfg.commands.weather_cache_path,
        reminders_path=cfg.commands.reminders_path,
        todo_path=cfg.commands.todo_path,
        screenshots_dir=cfg.commands.screenshots_dir,
        default_browser_url=cfg.commands.default_browser_url,
        **{"default_city": cfg.commands.default_city},
        allowed_apps=cfg.commands.allowed_apps,
        allow_shell=cfg.commands.allow_shell,
        allow_desktop_launch=cfg.commands.allow_desktop_launch,
    )

    reminders_lock = threading.Lock()

    def reminders_worker() -> None:
        path = cfg.commands.reminders_path
        while True:
            try:
                now = time.time()
                with reminders_lock:
                    if not path.exists():
                        time.sleep(1.0)
                        continue
                    data = json.loads(path.read_text(encoding="utf-8")) or {}
                    items = data.get("items") or []
                    changed = False
                    due_texts: list[str] = []
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        if it.get("fired"):
                            continue
                        try:
                            due_ts = float(it.get("due_ts", 0.0) or 0.0)
                        except Exception:
                            continue
                        if due_ts and due_ts <= now:
                            it["fired"] = True
                            changed = True
                            msg = str(it.get("text", "") or "").strip() or "напоминание"
                            due_texts.append(msg)
                    if changed:
                        tmp = path.with_suffix(path.suffix + ".tmp")
                        tmp.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        tmp.replace(path)
                for msg in due_texts:
                    ans = f"Напоминаю: {msg}."
                    status.update(state="speaking", message="Напоминание…", force=True)
                    ls, ss, vol = _tts_style(ans, cfg.tts.length_scale, cfg.tts.sentence_silence, cfg.tts.volume)
                    tts.speak(ans, length_scale=ls, sentence_silence=ss, volume=vol)
                    status.update(last_answer=ans, force=True)
                    status.update(state="idle", message="Готов")
            except Exception:
                pass
            time.sleep(1.0)

    try:
        stt = FasterWhisperSTT(
            model_size=cfg.stt.model_size,
            model_path=cfg.stt.model_path,
            device=cfg.stt.device,
            compute_type=cfg.stt.compute_type,
            cpu_threads=cfg.stt.cpu_threads,
            num_workers=cfg.stt.num_workers,
            language=cfg.stt.language,
            beam_size=cfg.stt.beam_size,
            vad_filter=cfg.stt.vad_filter,
            initial_prompt=cfg.stt.initial_prompt,
            min_avg_logprob=cfg.stt.min_avg_logprob,
            max_no_speech_prob=cfg.stt.max_no_speech_prob,
        )
    except Exception as e:
        status.update(state="error", message=str(e), force=True)
        raise

    tts = PiperTTS(
        voice=PiperVoice(model_path=cfg.tts.model_path, config_path=cfg.tts.config_path),
        output_device=cfg.audio.output_device,
        length_scale=cfg.tts.length_scale,
        sentence_silence=cfg.tts.sentence_silence,
        volume=cfg.tts.volume,
    )

    # фоновые напоминания
    threading.Thread(target=reminders_worker, daemon=True).start()

    ollama_fast: OllamaClient | None = None
    ollama_quality: OllamaClient | None = None
    llamacpp_llm: LlamaCppClient | None = None
    llm_warned = False
    llm_disabled_due_to_unreachable = False
    if cfg.llm.enabled:
        if cfg.llm.backend == "llamacpp":
            candidate_llama = LlamaCppClient(base_url=cfg.llm.llamacpp_base_url, timeout_s=cfg.llm.timeout_s)
            if candidate_llama.ping():
                llamacpp_llm = candidate_llama
            elif cfg.llm.disable_if_unreachable:
                llamacpp_llm = None
                llm_disabled_due_to_unreachable = True
                print("LLM отключен: llama.cpp server недоступен.")
            else:
                llamacpp_llm = candidate_llama
        else:
            # Ollama: один или два клиента (fast + quality)
            if cfg.llm.dual_model_enabled:
                candidate_fast = OllamaClient(base_url=cfg.llm.base_url, model=cfg.llm.fast_model, timeout_s=cfg.llm.timeout_s)
                candidate_quality = OllamaClient(base_url=cfg.llm.base_url, model=cfg.llm.quality_model, timeout_s=cfg.llm.timeout_s)
            else:
                candidate_fast = None
                candidate_quality = OllamaClient(base_url=cfg.llm.base_url, model=cfg.llm.model, timeout_s=cfg.llm.timeout_s)

            candidate = candidate_quality
        if (not candidate.ping()) and cfg.llm.autostart and shutil.which("ollama"):
            # попытка поднять Ollama автоматически
            try:
                # через systemd --user, если есть
                subprocess.run(["systemctl", "--user", "start", "ollama.service"], check=False)
            except Exception:
                pass
            # и fallback: прямой запуск
            try:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            # ждём несколько секунд
            for _ in range(20):
                if candidate.ping():
                    break
                time.sleep(0.2)

        if candidate_quality.ping():
            if not candidate_quality.has_model():
                if cfg.llm.auto_pull_model and shutil.which("ollama"):
                    status.update(state="thinking", message="Скачиваю LLM модель…", force=True)
                    subprocess.run(["ollama", "pull", cfg.llm.quality_model], check=False)
                    ollama_quality = candidate_quality if candidate_quality.has_model() else None
                else:
                    ollama_quality = None
                    print("LLM отключен: quality модель Ollama не скачана.")
            else:
                ollama_quality = candidate_quality
        elif cfg.llm.disable_if_unreachable:
            ollama_quality = None
            llm_disabled_due_to_unreachable = True
            print("LLM отключен: Ollama недоступна (localhost:11434).")
        else:
            ollama_quality = candidate_quality

        if cfg.llm.dual_model_enabled and candidate_fast is not None and candidate_fast.ping() and candidate_fast.has_model():
            ollama_fast = candidate_fast

    # Прогрев: чтобы первая реальная реплика отвечалась быстрее
    if cfg.llm.enabled and cfg.llm.warmup_on_start:
        warm_clients = [c for c in (ollama_fast, ollama_quality) if c is not None]
        if warm_clients:
            def _warm_all() -> None:
                for c in warm_clients:
                    try:
                        c.chat(
                            [ChatMessage(role="system", content="."), ChatMessage(role="user", content=".")],
                            num_predict=1,
                            temperature=0.0,
                            num_ctx=cfg.llm.ollama_num_ctx,
                            num_thread=cfg.llm.ollama_num_thread,
                            keep_alive=cfg.llm.ollama_keep_alive,
                        )
                    except Exception:
                        pass

            threading.Thread(target=_warm_all, name="jarvis-llm-warmup", daemon=True).start()

    detector = None
    if cfg.wake_word.enabled:
        try:
            from vosk import SetLogLevel

            SetLogLevel(-1)
        except Exception:
            pass
        detector = VoskWakeWordDetector(
            model_path=str(cfg.wake_word.vosk_model_path),
            sample_rate=cfg.audio.sample_rate,
            phrases=cfg.wake_word.phrases,
            input_device=cfg.audio.input_device,
            blocksize=cfg.wake_word.blocksize,
            min_rms=cfg.wake_word.min_rms,
        )

    history: list[ChatMessage] = [ChatMessage(role="system", content=SYSTEM_PROMPT)]

    last_llm_user: str | None = None
    last_llm_answer: str | None = None

    verbosity_mode: str = "normal"
    pending_action: str | None = None
    pending_action_until: float = 0.0
    last_heard: str = ""

    timer_lock = threading.Lock()
    timer_deadline: float | None = None

    def _timer_worker() -> None:
        nonlocal timer_deadline
        while True:
            time.sleep(0.2)
            with timer_lock:
                dl = timer_deadline
            if dl is None:
                continue
            if time.monotonic() >= dl:
                with timer_lock:
                    timer_deadline = None
                try:
                    status.update(state="speaking", message="Таймер")
                    s = "Таймер закончился."
                    ls, ss, vol = _tts_style(s, cfg.tts.length_scale, cfg.tts.sentence_silence, cfg.tts.volume)
                    tts.speak(s, length_scale=ls, sentence_silence=ss, volume=vol)
                    status.update(state="idle", message="Готов")
                except Exception:
                    pass

    threading.Thread(target=_timer_worker, name="jarvis-timer", daemon=True).start()

    print("Jarvis готов. Скажите ключевое слово…" if detector else "Jarvis готов. Говорите…")
    status.update(state="idle", message="Готов", force=True)

    try:
        while True:
            if detector:
                status.update(state="listening_wakeword", message="Слушаю ключевое слово…")

                def _lvl(x: float) -> None:
                    status.update(mic_level=x)

                ww = detector.listen(on_mic_level=_lvl)
                print(f"Активация: {ww.phrase}")
                if cfg.audio.activation_feedback == "beep":
                    beep(cfg.audio.sample_rate, output_device=cfg.audio.output_device)
                elif cfg.audio.activation_feedback == "tts":
                    s = cfg.audio.activation_text
                    ls, ss, vol = _tts_style(s, cfg.tts.length_scale, cfg.tts.sentence_silence, cfg.tts.volume)
                    tts.speak(s, length_scale=ls, sentence_silence=ss, volume=vol)
                    if getattr(tts, "last_error", ""):
                        status.update(last_error=f"TTS: {tts.last_error}", force=True)
                time.sleep(float(cfg.audio.activation_delay_s))

                start_immediately = bool(cfg.recording.start_immediately_after_wakeword)
            else:
                start_immediately = False

            status.update(state="recording", message="Говорите…")

            def _rec_lvl(x: float) -> None:
                thr_now = float(getattr(record_utterance, "last_threshold", cfg.recording.start_threshold))
                gain_now = float(getattr(record_utterance, "last_gain", 1.0))
                status.update(mic_level=x, recording_threshold=thr_now, recording_gain=gain_now)

            recorded = record_utterance(
                sample_rate=cfg.audio.sample_rate,
                input_device=cfg.audio.input_device,
                max_s=cfg.recording.max_s,
                start_threshold=cfg.recording.start_threshold,
                start_immediately=start_immediately,
                pre_roll_s=cfg.recording.pre_roll_s,
                auto_threshold=cfg.recording.auto_threshold,
                noise_calibrate_s=cfg.recording.noise_calibrate_s,
                threshold_multiplier=cfg.recording.threshold_multiplier,
                end_silence_s=cfg.recording.end_silence_s,
                chunk_ms=cfg.recording.chunk_ms,
                on_rms=_rec_lvl,
                auto_gain=cfg.recording.auto_gain,
                target_rms=cfg.recording.target_rms,
                max_gain=cfg.recording.max_gain,
            )

            # последний rms от record_utterance
            thr = float(getattr(record_utterance, "last_threshold", cfg.recording.start_threshold))
            gain = float(getattr(record_utterance, "last_gain", 1.0))
            status.update(
                mic_level=float(getattr(record_utterance, "last_rms", 0.0)),
                recording_threshold=thr,
                recording_gain=gain,
            )

            status.update(state="transcribing", message="Распознаю…")
            stt_res = stt.transcribe_pcm16(recorded.pcm16, recorded.sample_rate)
            text = stt_res.text.strip()
            if not text:
                started = bool(getattr(record_utterance, "last_started", False))
                rms = float(getattr(record_utterance, "last_rms", 0.0))
                msg = (
                    f"Не расслышал (rms {rms:.3f}, порог {thr:.3f})"
                    if started
                    else f"Не услышал речь (rms {rms:.3f}, порог {thr:.3f})"
                )
                print(msg)
                status.update(state="idle", message=msg, force=True)
                continue

            text = re.sub(r"^(джарвис|ассистент)[\s,.:;!?-]+", "", text, flags=re.IGNORECASE).strip()

            print(f"Вы: {text}")
            last_heard = text
            status.update(last_heard=text, force=True)

            # --- Подтверждение опасных действий ---
            t_low = text.strip().lower()
            yes = {"да", "давай", "подтверждаю", "ок", "окей"}
            no = {"нет", "отмена", "не надо"}
            if pending_action and (time.monotonic() < pending_action_until):
                if t_low in no:
                    pending_action = None
                    answer = "Ок. Отменил."
                    print(f"Jarvis: {answer}")
                    status.update(state="speaking", message="Отвечаю…")
                    ls, ss, vol = _tts_style(answer, cfg.tts.length_scale, cfg.tts.sentence_silence, cfg.tts.volume)
                    tts.speak(answer, length_scale=ls, sentence_silence=ss, volume=vol)
                    status.update(last_answer=answer, force=True)
                    status.update(state="idle", message="Готов")
                    continue
                if t_low in yes or t_low.startswith("да,") or t_low.startswith("да "):
                    action = pending_action
                    pending_action = None
                    answer = _execute_system_action(action)
                    print(f"Jarvis: {answer}")
                    status.update(state="speaking", message="Выполняю…")
                    ls, ss, vol = _tts_style(answer, cfg.tts.length_scale, cfg.tts.sentence_silence, cfg.tts.volume)
                    tts.speak(answer, length_scale=ls, sentence_silence=ss, volume=vol)
                    status.update(last_answer=answer, force=True)
                    status.update(state="idle", message="Готов")
                    continue

            cmd = router.handle(text)
            if cmd is not None:
                if isinstance(cmd, TimerSetResponse):
                    with timer_lock:
                        timer_deadline = time.monotonic() + float(cmd.seconds)
                    answer = cmd.text
                elif isinstance(cmd, TimerCancelResponse):
                    with timer_lock:
                        timer_deadline = None
                    answer = cmd.text
                elif isinstance(cmd, TimerStatusRequest):
                    with timer_lock:
                        dl = timer_deadline
                    if dl is None:
                        answer = "Таймер не запущен."
                    else:
                        rem = max(0.0, dl - time.monotonic())
                        mins = int(rem // 60)
                        secs = int(round(rem % 60))
                        answer = f"Осталось примерно {mins} минут {secs} секунд." if mins else f"Осталось примерно {secs} секунд."
                elif isinstance(cmd, VerbositySetResponse):
                    verbosity_mode = cmd.mode
                    answer = cmd.text
                elif isinstance(cmd, SpeakStatusRequest):
                    answer = _format_status_for_speech(
                        last_heard=last_heard,
                        llm_enabled=(ollama_quality is not None or llamacpp_llm is not None),
                        verbosity_mode=verbosity_mode,
                    )
                elif isinstance(cmd, RepeatLastHeardRequest):
                    answer = f"Ты сказал: {last_heard}." if last_heard else "Я пока ничего не услышал."
                elif isinstance(cmd, PendingSystemActionResponse):
                    pending_action = cmd.action
                    pending_action_until = time.monotonic() + 8.0
                    answer = cmd.text
                else:
                    answer = cmd.text
            elif (ollama_quality is not None) or (llamacpp_llm is not None):
                status.update(state="thinking", message="Думаю…")
                print("Думаю…")
                history.append(ChatMessage(role="user", content=text))
                try:
                    if llamacpp_llm is not None:
                        if not llamacpp_llm.ping():
                            raise LlamaCppUnavailable("llama.cpp недоступен")
                        msgs = [LlamaMessage(role=m.role, content=m.content) for m in history]
                        answer = llamacpp_llm.chat(msgs, max_tokens=cfg.llm.num_predict, temperature=cfg.llm.temperature)
                    else:
                        # выбор модели
                        t = text.strip().lower()
                        escalate_words = (
                            "подробнее",
                            "разверни",
                            "поясни",
                            "уточни",
                            "детальнее",
                            "продолжай",
                            "а почему",
                        )
                        wants_details = t == "подробнее" or t.startswith("подробнее ") or any(w in t for w in escalate_words)

                        # Fast-first: по умолчанию спрашиваем быструю модель, если она доступна.
                        selected = ollama_fast if (cfg.llm.dual_model_enabled and ollama_fast is not None) else ollama_quality

                        # Escalation: на "подробнее/уточни" идём в quality и даём короткий контекст (прошлый Q/A)
                        llm_messages = [history[0], ChatMessage(role="user", content=text)]
                        if wants_details and ollama_quality is not None and last_llm_user and last_llm_answer:
                            selected = ollama_quality
                            llm_messages = [
                                history[0],
                                ChatMessage(role="user", content=last_llm_user),
                                ChatMessage(role="assistant", content=last_llm_answer),
                                ChatMessage(role="user", content="Раскрой подробнее предыдущий ответ. Дай 3–5 предложений."),
                            ]

                        assert selected is not None
                        if not selected.ping():
                            raise OllamaUnavailable("Ollama недоступна")

                        # лимиты зависят от режима "короче/нормально/подробнее"
                        max_sentences, max_chars, num_predict = _llm_limits(verbosity_mode, cfg.llm.fast_max_sentences, cfg.llm.fast_max_chars, cfg.llm.num_predict)

                        if cfg.llm.fast_mode:
                            answer = selected.chat_fast(
                                llm_messages,
                                num_predict=num_predict,
                                temperature=cfg.llm.temperature,
                                num_ctx=cfg.llm.ollama_num_ctx,
                                num_thread=cfg.llm.ollama_num_thread,
                                keep_alive=cfg.llm.ollama_keep_alive,
                                max_sentences=max_sentences,
                                max_chars=max_chars,
                            )
                        else:
                            answer = selected.chat(
                                llm_messages,
                                num_predict=num_predict,
                                temperature=cfg.llm.temperature,
                                num_ctx=cfg.llm.ollama_num_ctx,
                                num_thread=cfg.llm.ollama_num_thread,
                                keep_alive=cfg.llm.ollama_keep_alive,
                            )

                        # сохраняем краткий контекст для "подробнее"
                        if not wants_details:
                            last_llm_user = text
                            last_llm_answer = answer
                    history.append(ChatMessage(role="assistant", content=answer))
                    history[:] = history[:1] + history[-int(cfg.llm.history_max_messages) :]
                except (OllamaUnavailable, LlamaCppUnavailable) as e:
                    # В рантайме НЕ качаем модель и не делаем долгих повторов.
                    if not llm_warned:
                        llm_warned = True
                        s = "Нейросеть сейчас недоступна. Отвечаю командами."
                        ls, ss, vol = _tts_style(s, cfg.tts.length_scale, cfg.tts.sentence_silence, cfg.tts.volume)
                        tts.speak(s, length_scale=ls, sentence_silence=ss, volume=vol)
                    print(f"LLM недоступна: {e}")
                    ollama_fast = None
                    ollama_quality = None
                    llamacpp_llm = None
                    answer = "Нейросеть сейчас недоступна."
            else:
                if llm_disabled_due_to_unreachable:
                    answer = "Я сейчас не вижу Ollama (localhost:11434). Запусти Ollama или выключи LLM в трее — и попробуй ещё раз."
                else:
                    answer = "Я пока умею только простые команды."

            print(f"Jarvis: {answer}")
            status.update(state="speaking", message="Отвечаю…")
            ls, ss, vol = _tts_style(answer, cfg.tts.length_scale, cfg.tts.sentence_silence, cfg.tts.volume)
            tts.speak(answer, length_scale=ls, sentence_silence=ss, volume=vol)
            if getattr(tts, "last_error", ""):
                status.update(last_error=f"TTS: {tts.last_error}", force=True)
            status.update(last_answer=answer, force=True)
            status.update(state="idle", message="Готов")

    except KeyboardInterrupt:
        print("\nОстановлено.")
        status.update(state="idle", message="Остановлено", force=True)
        return
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        status.update(state="error", message=str(e), force=True)
        raise


def _llm_limits(mode: str, base_sentences: int, base_chars: int, base_num_predict: int) -> tuple[int, int, int]:
    m = (mode or "normal").strip().lower()
    if m == "short":
        return 1, min(220, int(base_chars)), min(64, int(base_num_predict))
    if m == "long":
        return max(3, int(base_sentences) + 2), max(420, int(base_chars) * 2), max(128, int(base_num_predict) * 2)
    return int(base_sentences), int(base_chars), int(base_num_predict)


def _execute_system_action(action: str) -> str:
    action = (action or "").strip().lower()
    if action == "lock":
        # best-effort
        for cmd in (["loginctl", "lock-session"], ["gnome-screensaver-command", "-l"]):
            try:
                subprocess.run(cmd, check=True, timeout=3.0)
                return "Блокирую."
            except Exception:
                continue
        return "Не смог заблокировать экран."

    if action == "suspend":
        cmd = ["systemctl", "suspend"]
        try:
            subprocess.run(cmd, check=True, timeout=3.0)
            return "Ок. Ухожу в сон."
        except Exception:
            return "Не смог отправить в сон (нужны права/политики)."

    if action == "reboot":
        cmd = ["systemctl", "reboot"]
        try:
            subprocess.run(cmd, check=True, timeout=3.0)
            return "Перезагружаю."
        except Exception:
            return "Не смог перезагрузить (нужны права/политики)."

    if action == "poweroff":
        cmd = ["systemctl", "poweroff"]
        try:
            subprocess.run(cmd, check=True, timeout=3.0)
            return "Выключаю."
        except Exception:
            return "Не смог выключить (нужны права/политики)."

    return "Не понял действие."


def _format_status_for_speech(*, last_heard: str, llm_enabled: bool, verbosity_mode: str) -> str:
    parts: list[str] = []
    parts.append("Я на месте.")
    parts.append("Нейросеть включена" if llm_enabled else "Нейросеть выключена")
    if verbosity_mode:
        parts.append(f"режим {verbosity_mode}")
    if last_heard:
        parts.append(f"последнее: {last_heard}")
    return ", ".join(parts) + "."


def _tts_style(text: str, base_length_scale: float, base_sentence_silence: float, base_volume: float) -> tuple[float, float, float]:
    t = (text or "").strip()
    low = t.lower()
    ls = float(base_length_scale)
    ss = float(base_sentence_silence)
    vol = float(base_volume)

    is_error = any(w in low for w in ("не смог", "ошибка", "недоступ", "таймаут"))
    is_question = t.endswith("?")
    is_short = len(t) <= 18

    if is_error:
        ls *= 0.98
        ss *= 1.15
        vol *= 0.92
    elif is_question:
        ls *= 1.02
        ss *= 1.25
        vol *= 1.00
    elif is_short:
        ls *= 0.92
        ss *= 0.90
        vol *= 1.05
    else:
        # лёгкая вариация: чем больше знаков пунктуации, тем больше пауз
        punct = sum(1 for ch in t if ch in ",;:")
        ss *= 1.0 + min(0.35, punct * 0.08)

    # безопасные диапазоны
    ls = max(0.70, min(1.15, ls))
    ss = max(0.05, min(0.35, ss))
    vol = max(0.25, min(1.2, vol))
    return ls, ss, vol
