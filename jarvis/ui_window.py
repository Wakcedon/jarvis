from __future__ import annotations

import os
import subprocess
import sys
import threading
import json
from dataclasses import dataclass
from pathlib import Path

from jarvis.paths import find_config_path, get_paths, get_root
from jarvis.presets import apply_room_preset, apply_speed_preset
from jarvis.yaml_config import get_dotted, set_dotted


@dataclass
class ServiceBackend:
    mode: str  # systemd|portable
    pid_file: Path


def _has_systemd_user() -> bool:
    try:
        r = subprocess.run(["systemctl", "--user", "is-enabled", "jarvis.service"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode in (0, 1, 3)
    except Exception:
        return False


def _systemctl_user(*args: str) -> int:
    try:
        r = subprocess.run(["systemctl", "--user", *args], check=False)
        return int(r.returncode)
    except Exception:
        return 1


def _portable_start(cfg: Path, pid_file: Path) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Keep paths consistent with UI and XDG.
    try:
        env.setdefault("JARVIS_ROOT", str(get_root()))
        env.setdefault("JARVIS_STATUS", str(get_paths().status_path))
    except Exception:
        pass
    p = subprocess.Popen([sys.executable, "-m", "jarvis", "run", "--config", str(cfg)], env=env)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(int(p.pid)), encoding="utf-8")


def _portable_stop(pid_file: Path) -> None:
    try:
        if not pid_file.exists():
            return
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return
    try:
        os.kill(pid, 15)
    except Exception:
        pass


def _read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _status_ts(st: dict) -> float:
    try:
        return float(st.get("ts", 0.0) or 0.0)
    except Exception:
        return 0.0


def _pick_status_path(*, xdg_path: Path, runtime_path: Path) -> Path:
    """Pick the status file that exists and looks freshest (ts/mtime)."""

    best = xdg_path
    best_score = -1.0
    for p in (xdg_path, runtime_path):
        if not p.exists():
            continue
        st = _read_status(p)
        try:
            mtime = float(p.stat().st_mtime)
        except Exception:
            mtime = 0.0
        ts = _status_ts(st)
        # Prefer filesystem mtime (reliable) and use ts only as a tie-breaker.
        score = mtime + (min(0.999, max(0.0, (ts - mtime))) if ts else 0.0)
        if score > best_score:
            best_score = score
            best = p
    return best


def _tail_lines(path: Path, *, max_lines: int = 400) -> list[str]:
    try:
        if not path.exists():
            return []
        data = path.read_text(encoding="utf-8", errors="ignore")
        lines = data.splitlines()[-int(max_lines) :]
        return lines
    except Exception:
        return []


def _make_ui_class():
    # Preferred UI: modern dark theme via CustomTkinter.
    try:
        import customtkinter as ctk
        import sounddevice as sd

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        class JarvisUICTk(ctk.CTk):
            def __init__(self, *, config_path: Path) -> None:
                super().__init__()
                self.title("Jarvis")
                self.geometry("980x620")
                self.minsize(900, 560)

                self._paths = get_paths()
                self._status_xdg = self._paths.status_path
                self._status_runtime = get_root() / "runtime" / "status.json"
                self._status_path = _pick_status_path(xdg_path=self._status_xdg, runtime_path=self._status_runtime)
                self._log_path = self._paths.state_dir / "jarvis.log"

                self._config_path = config_path
                self._template_path = (Path(__file__).resolve().parents[1] / "config.yaml").resolve()
                self._svc = ServiceBackend(
                    mode="systemd" if _has_systemd_user() else "portable",
                    pid_file=Path.home() / ".local/state/jarvis/ui.pid",
                )

                self._mic_map: dict[str, int | None] = {"Авто": None}
                try:
                    devices = sd.query_devices()
                except Exception:
                    devices = []
                for idx, dev in enumerate(devices):
                    if int(dev.get("max_input_channels", 0) or 0) <= 0:
                        continue
                    self._mic_map[f"{idx}: {dev.get('name','?')}"] = int(idx)

                self._log_pos = 0
                self._log_size = 0

                self._build(ctk)
                # preload logs (tail)
                try:
                    if self._log_path.exists():
                        tail = "\n".join(_tail_lines(self._log_path, max_lines=200))
                        if tail:
                            self.log_box.configure(state="normal")
                            self.log_box.insert("end", tail + "\n")
                            self.log_box.see("end")
                            self.log_box.configure(state="disabled")
                        self._log_pos = int(self._log_path.stat().st_size)
                except Exception:
                    pass
                self._reload()
                self._poll_status()
                self._poll_logs()

            def _build(self, ctk) -> None:
                self.grid_rowconfigure(0, weight=1)
                self.grid_columnconfigure(1, weight=1)

                self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
                self.sidebar.grid(row=0, column=0, sticky="nsew")
                self.sidebar.grid_rowconfigure(20, weight=1)

                self.main = ctk.CTkFrame(self, corner_radius=0)
                self.main.grid(row=0, column=1, sticky="nsew")
                self.main.grid_rowconfigure(2, weight=1)
                self.main.grid_columnconfigure(0, weight=1)

                # --- Sidebar ---
                ctk.CTkLabel(self.sidebar, text="Jarvis", font=("Segoe UI", 20, "bold")).grid(
                    row=0, column=0, padx=16, pady=(16, 4), sticky="w"
                )
                self.mode_label = ctk.CTkLabel(self.sidebar, text="", font=("Segoe UI", 12))
                self.mode_label.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

                btn_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
                btn_row.grid(row=2, column=0, padx=14, pady=(0, 10), sticky="ew")
                btn_row.grid_columnconfigure((0, 1, 2), weight=1)

                ctk.CTkButton(btn_row, text="Запуск", command=self._start).grid(row=0, column=0, padx=(0, 6), sticky="ew")
                ctk.CTkButton(btn_row, text="Стоп", command=self._stop).grid(row=0, column=1, padx=6, sticky="ew")
                ctk.CTkButton(btn_row, text="Рестарт", command=self._restart).grid(row=0, column=2, padx=(6, 0), sticky="ew")

                ctk.CTkLabel(self.sidebar, text="Пресеты", font=("Segoe UI", 12, "bold")).grid(
                    row=3, column=0, padx=16, pady=(8, 6), sticky="w"
                )
                preset_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
                preset_row.grid(row=4, column=0, padx=14, pady=(0, 10), sticky="ew")
                preset_row.grid_columnconfigure((0, 1), weight=1)
                ctk.CTkButton(preset_row, text="Скорость", command=lambda: self._apply_preset("speed")).grid(
                    row=0, column=0, padx=(0, 6), sticky="ew"
                )
                ctk.CTkButton(preset_row, text="Комната", command=lambda: self._apply_preset("room")).grid(
                    row=0, column=1, padx=(6, 0), sticky="ew"
                )

                ctk.CTkLabel(self.sidebar, text="Переключатели", font=("Segoe UI", 12, "bold")).grid(
                    row=5, column=0, padx=16, pady=(10, 6), sticky="w"
                )
                self.sw_wake = ctk.CTkSwitch(self.sidebar, text="Ключевое слово", command=self._toggle_wake)
                self.sw_wake.grid(row=6, column=0, padx=16, pady=6, sticky="w")
                self.sw_llm = ctk.CTkSwitch(self.sidebar, text="Нейросеть (Ollama)", command=self._toggle_llm)
                self.sw_llm.grid(row=7, column=0, padx=16, pady=6, sticky="w")
                self.sw_barge = ctk.CTkSwitch(self.sidebar, text="Перебить голосом", command=self._toggle_barge)
                self.sw_barge.grid(row=8, column=0, padx=16, pady=6, sticky="w")

                ctk.CTkLabel(self.sidebar, text="Микрофон", font=("Segoe UI", 12, "bold")).grid(
                    row=9, column=0, padx=16, pady=(12, 6), sticky="w"
                )
                self.mic_choice = ctk.StringVar(value="Авто")
                self.mic_menu = ctk.CTkOptionMenu(
                    self.sidebar,
                    values=list(self._mic_map.keys()),
                    variable=self.mic_choice,
                )
                self.mic_menu.grid(row=10, column=0, padx=14, pady=(0, 8), sticky="ew")
                ctk.CTkButton(self.sidebar, text="Применить", command=self._apply_mic).grid(
                    row=11, column=0, padx=14, pady=(0, 12), sticky="ew"
                )

                ctk.CTkButton(self.sidebar, text="Открыть config.yaml", command=self._open_config).grid(
                    row=21, column=0, padx=14, pady=(10, 8), sticky="ew"
                )
                ctk.CTkButton(self.sidebar, text="Обновить", command=self._reload).grid(
                    row=22, column=0, padx=14, pady=(0, 16), sticky="ew"
                )

                # --- Main ---
                header = ctk.CTkFrame(self.main)
                header.grid(row=0, column=0, padx=14, pady=(14, 10), sticky="ew")
                header.grid_columnconfigure(0, weight=1)

                self.lbl_state = ctk.CTkLabel(header, text="нет связи", font=("Segoe UI", 16, "bold"))
                self.lbl_state.grid(row=0, column=0, padx=12, pady=(10, 0), sticky="w")
                self.lbl_msg = ctk.CTkLabel(header, text="", font=("Segoe UI", 12))
                self.lbl_msg.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

                microw = ctk.CTkFrame(self.main)
                microw.grid(row=1, column=0, padx=14, pady=(0, 10), sticky="ew")
                microw.grid_columnconfigure(1, weight=1)
                ctk.CTkLabel(microw, text="Уровень микрофона", font=("Segoe UI", 12, "bold")).grid(
                    row=0, column=0, padx=12, pady=10, sticky="w"
                )
                self.mic_pct = ctk.CTkLabel(microw, text="0%", font=("Segoe UI", 12))
                self.mic_pct.grid(row=0, column=2, padx=12, pady=10, sticky="e")
                self.mic_bar = ctk.CTkProgressBar(microw)
                self.mic_bar.grid(row=0, column=1, padx=12, pady=12, sticky="ew")
                self.mic_bar.set(0.0)

                self.lat_label = ctk.CTkLabel(self.main, text="", font=("Segoe UI", 11))
                self.lat_label.grid(row=2, column=0, padx=18, pady=(0, 6), sticky="w")

                self.tabs = ctk.CTkTabview(self.main)
                self.tabs.grid(row=3, column=0, padx=14, pady=(0, 14), sticky="nsew")
                self.tabs.add("Логи")
                self.tabs.add("Детали")

                # Logs
                logs_tab = self.tabs.tab("Логи")
                logs_tab.grid_rowconfigure(1, weight=1)
                logs_tab.grid_columnconfigure(0, weight=1)
                self.log_path_lbl = ctk.CTkLabel(logs_tab, text=str(self._log_path), font=("Segoe UI", 11))
                self.log_path_lbl.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="w")
                self.log_box = ctk.CTkTextbox(logs_tab, wrap="none")
                self.log_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
                self.log_box.configure(state="disabled")

                # Details
                det_tab = self.tabs.tab("Детали")
                det_tab.grid_rowconfigure((1, 3), weight=1)
                det_tab.grid_columnconfigure(0, weight=1)
                ctk.CTkLabel(det_tab, text="Последнее распознано", font=("Segoe UI", 12, "bold")).grid(
                    row=0, column=0, padx=10, pady=(10, 6), sticky="w"
                )
                self.last_heard = ctk.CTkTextbox(det_tab, height=100, wrap="word")
                self.last_heard.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
                self.last_heard.configure(state="disabled")
                ctk.CTkLabel(det_tab, text="Последняя ошибка", font=("Segoe UI", 12, "bold")).grid(
                    row=2, column=0, padx=10, pady=(0, 6), sticky="w"
                )
                self.last_error = ctk.CTkTextbox(det_tab, height=100, wrap="word")
                self.last_error.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")
                self.last_error.configure(state="disabled")

            def _set_textbox(self, tb, text: str) -> None:
                try:
                    tb.configure(state="normal")
                    tb.delete("1.0", "end")
                    tb.insert("end", text)
                    tb.configure(state="disabled")
                except Exception:
                    pass

            def _reload(self) -> None:
                self.mode_label.configure(text=f"режим: {self._svc.mode}")
                self.sw_wake.select() if bool(get_dotted(self._config_path, "wake_word.enabled", True)) else self.sw_wake.deselect()
                self.sw_llm.select() if bool(get_dotted(self._config_path, "llm.enabled", True)) else self.sw_llm.deselect()
                self.sw_barge.select() if bool(get_dotted(self._config_path, "audio.barge_in_enabled", True)) else self.sw_barge.deselect()

                cur_dev = get_dotted(self._config_path, "audio.input_device", None)
                choice = "Авто"
                for k, v in self._mic_map.items():
                    if v == cur_dev:
                        choice = k
                        break
                self.mic_choice.set(choice)

            def _toggle_wake(self) -> None:
                set_dotted(self._config_path, "wake_word.enabled", "true" if self.sw_wake.get() else "false")

            def _toggle_llm(self) -> None:
                set_dotted(self._config_path, "llm.enabled", "true" if self.sw_llm.get() else "false")

            def _toggle_barge(self) -> None:
                set_dotted(self._config_path, "audio.barge_in_enabled", "true" if self.sw_barge.get() else "false")

            def _apply_mic(self) -> None:
                v = self._mic_map.get(self.mic_choice.get(), None)
                set_dotted(self._config_path, "audio.input_device", "null" if v is None else str(int(v)))
                self._reload()

            def _apply_preset(self, preset: str) -> None:
                try:
                    if preset == "speed":
                        apply_speed_preset(self._config_path, template_path=self._template_path)
                    else:
                        apply_room_preset(self._config_path, template_path=self._template_path)
                    self._reload()
                except Exception as e:
                    try:
                        self.lbl_msg.configure(text=f"Ошибка: {e}")
                    except Exception:
                        pass

            def _start(self) -> None:
                if self._svc.mode == "systemd":
                    _systemctl_user("start", "jarvis.service")
                    return
                threading.Thread(target=_portable_start, args=(self._config_path, self._svc.pid_file), daemon=True).start()

            def _stop(self) -> None:
                if self._svc.mode == "systemd":
                    _systemctl_user("stop", "jarvis.service")
                    return
                _portable_stop(self._svc.pid_file)

            def _restart(self) -> None:
                if self._svc.mode == "systemd":
                    _systemctl_user("restart", "jarvis.service")
                    return
                self._stop()
                self._start()

            def _open_config(self) -> None:
                subprocess.Popen(["xdg-open", str(self._config_path)])

            def _poll_status(self) -> None:
                # Choose freshest status source on each poll (XDG vs runtime).
                self._status_path = _pick_status_path(xdg_path=self._status_xdg, runtime_path=self._status_runtime)
                st = _read_status(self._status_path)
                raw_state = str(st.get("state", "нет связи"))
                state_map = {
                    "idle": "Готов",
                    "starting": "Запуск",
                    "listening": "Слушаю",
                    "thinking": "Думаю",
                    "speaking": "Говорю",
                    "error": "Ошибка",
                }
                state = state_map.get(raw_state, raw_state)
                msg = str(st.get("message", ""))
                lvl = float(st.get("mic_level", 0.0) or 0.0)
                pct = int(max(0.0, min(1.0, lvl / 0.2)) * 100)

                # Staleness hint.
                try:
                    import time as _time

                    ts = _status_ts(st)
                    try:
                        mtime = float(self._status_path.stat().st_mtime)
                    except Exception:
                        mtime = 0.0
                    ref = ts or mtime
                    age_s = max(0.0, float(_time.time()) - ref) if ref else 0.0
                    try:
                        self.mode_label.configure(
                            text=f"режим: {self._svc.mode}\nstatus: {self._status_path}\nвозраст: {age_s:.0f}с"
                        )
                    except Exception:
                        pass
                    if (not st) and (not self._status_path.exists()):
                        msg = f"status.json не найден: {self._status_path}"
                    elif age_s > 6.0 and (raw_state in ("starting", "idle")):
                        msg = (msg + " | ") if msg else ""
                        msg += f"последнее обновление {age_s:.0f}с назад"
                except Exception:
                    pass

                self.lbl_state.configure(text=state)
                self.lbl_msg.configure(text=msg)
                self.mic_pct.configure(text=f"{pct}%")
                try:
                    self.mic_bar.set(pct / 100.0)
                except Exception:
                    pass

                heard = str(st.get("last_heard", "") or "")
                err = str(st.get("last_error", "") or "")
                self._set_textbox(self.last_heard, heard if heard else "—")
                self._set_textbox(self.last_error, err if err else "—")

                lr = float(st.get("latency_record_ms", 0.0) or 0.0)
                ls = float(st.get("latency_stt_ms", 0.0) or 0.0)
                ll = float(st.get("latency_llm_ms", 0.0) or 0.0)
                lt = float(st.get("latency_tts_ms", 0.0) or 0.0)
                ltot = float(st.get("latency_total_ms", 0.0) or 0.0)
                self.lat_label.configure(text=f"Задержки (мс): запись={lr:.0f} stt={ls:.0f} llm={ll:.0f} tts={lt:.0f} всего={ltot:.0f}")

                self.after(200, self._poll_status)

            def _poll_logs(self) -> None:
                try:
                    if not self._log_path.exists():
                        self.after(500, self._poll_logs)
                        return
                    st = self._log_path.stat()
                    size = int(st.st_size)
                    if size < self._log_pos:
                        self._log_pos = 0

                    if size > self._log_pos:
                        with self._log_path.open("r", encoding="utf-8", errors="ignore") as f:
                            f.seek(self._log_pos)
                            chunk = f.read(max(64_000, size - self._log_pos))
                            self._log_pos = f.tell()

                        if chunk:
                            self.log_box.configure(state="normal")
                            self.log_box.insert("end", chunk)
                            # keep last N chars
                            if float(self.log_box.index("end").split(".")[0]) > 2000:
                                self.log_box.delete("1.0", "end-1500l")
                            self.log_box.see("end")
                            self.log_box.configure(state="disabled")
                except Exception:
                    pass

                self.after(500, self._poll_logs)

        return JarvisUICTk
    except Exception:
        # Fall back to ttk UI below.
        pass

    import tkinter as tk
    from tkinter import ttk, messagebox

    import sounddevice as sd

    class JarvisUI(tk.Tk):
        def __init__(self, *, config_path: Path) -> None:
            super().__init__()
            self.title("Jarvis")
            self.geometry("760x560")

            self._paths = get_paths()
            self._status_path = self._paths.status_path
            # portable fallback for status
            if not self._status_path.exists():
                self._status_path = get_root() / "runtime" / "status.json"
            self._log_path = self._paths.state_dir / "jarvis.log"

            self._config_path = config_path
            self._template_path = (Path(__file__).resolve().parents[1] / "config.yaml").resolve()
            self._svc = ServiceBackend(
                mode="systemd" if _has_systemd_user() else "portable",
                pid_file=Path.home() / ".local/state/jarvis/ui.pid",
            )

            self._apply_dark_theme(ttk)

            self._build()
            self._reload()
            self._poll()

        def _apply_dark_theme(self, ttk) -> None:
            try:
                style = ttk.Style(self)
                style.theme_use("clam")
                bg = "#1f1f1f"
                fg = "#e6e6e6"
                dim = "#2a2a2a"
                accent = "#3a7afe"

                self.configure(bg=bg)
                style.configure("TFrame", background=bg)
                style.configure("TLabel", background=bg, foreground=fg)
                style.configure("TLabelframe", background=bg, foreground=fg)
                style.configure("TLabelframe.Label", background=bg, foreground=fg)
                style.configure("TButton", background=dim, foreground=fg, padding=6)
                style.map("TButton", background=[("active", "#333333")])
                style.configure("TCheckbutton", background=bg, foreground=fg)
                style.configure("TCombobox", fieldbackground=dim, background=dim, foreground=fg)
                style.configure("TNotebook", background=bg, borderwidth=0)
                style.configure("TNotebook.Tab", background=dim, foreground=fg, padding=(10, 6))
                style.map("TNotebook.Tab", background=[("selected", "#333333")])
                style.configure("Horizontal.TProgressbar", troughcolor=dim, background=accent)
            except Exception:
                pass

        def _build(self) -> None:
            root = ttk.Frame(self, padding=12)
            root.pack(fill=tk.BOTH, expand=True)

            nb = ttk.Notebook(root)
            nb.pack(fill=tk.BOTH, expand=True)

            tab_status = ttk.Frame(nb)
            tab_logs = ttk.Frame(nb)
            nb.add(tab_status, text="Статус")
            nb.add(tab_logs, text="Логи")

            self._build_status(tab_status, tk, ttk)
            self._build_logs(tab_logs, tk, ttk)

        def _build_status(self, root, tk, ttk) -> None:
            top = ttk.Frame(root)
            top.pack(fill=tk.X)

            # Presets
            preset_box = ttk.LabelFrame(top, text="Пресет")
            preset_box.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.preset = tk.StringVar(value="room")
            ttk.Combobox(preset_box, textvariable=self.preset, values=["speed", "room"], state="readonly").pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8), pady=8
            )
            ttk.Button(preset_box, text="Применить", command=self._apply_preset).pack(side=tk.RIGHT, padx=8, pady=8)

            svc_box = ttk.LabelFrame(top, text="Jarvis")
            svc_box.pack(side=tk.RIGHT, fill=tk.X, padx=(12, 0))
            ttk.Button(svc_box, text="Start", command=self._start).pack(side=tk.LEFT, padx=6, pady=8)
            ttk.Button(svc_box, text="Stop", command=self._stop).pack(side=tk.LEFT, padx=6, pady=8)
            ttk.Button(svc_box, text="Restart", command=self._restart).pack(side=tk.LEFT, padx=6, pady=8)

            # Live status
            live = ttk.LabelFrame(root, text="Мониторинг")
            live.pack(fill=tk.X, pady=(12, 0))

            self._st_state = tk.StringVar(value="—")
            self._st_msg = tk.StringVar(value="")
            self._st_last = tk.StringVar(value="")
            self._st_err = tk.StringVar(value="")
            self._st_lat = tk.StringVar(value="")

            row1 = ttk.Frame(live)
            row1.pack(fill=tk.X, padx=8, pady=(8, 0))
            ttk.Label(row1, text="State:").pack(side=tk.LEFT)
            ttk.Label(row1, textvariable=self._st_state).pack(side=tk.LEFT, padx=(6, 0))
            ttk.Label(row1, textvariable=self._st_msg).pack(side=tk.RIGHT)

            row2 = ttk.Frame(live)
            row2.pack(fill=tk.X, padx=8, pady=6)
            ttk.Label(row2, text="Mic:").pack(side=tk.LEFT)
            self._mic_pct = tk.StringVar(value="0%")
            ttk.Label(row2, textvariable=self._mic_pct).pack(side=tk.LEFT, padx=(6, 10))
            self._mic_bar = ttk.Progressbar(row2, mode="determinate", maximum=100)
            self._mic_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

            row3 = ttk.Frame(live)
            row3.pack(fill=tk.X, padx=8, pady=(0, 8))
            ttk.Label(row3, textvariable=self._st_lat).pack(side=tk.LEFT)

            last_box = ttk.LabelFrame(root, text="Последнее")
            last_box.pack(fill=tk.X, pady=(12, 0))
            ttk.Label(last_box, textvariable=self._st_last, wraplength=700).pack(anchor="w", padx=8, pady=8)

            err_box = ttk.LabelFrame(root, text="Ошибка")
            err_box.pack(fill=tk.X, pady=(12, 0))
            ttk.Label(err_box, textvariable=self._st_err, wraplength=700).pack(anchor="w", padx=8, pady=8)

            # Toggles
            toggles = ttk.LabelFrame(root, text="Быстрые переключатели")
            toggles.pack(fill=tk.X, pady=(12, 0))

            self.wake_enabled = tk.BooleanVar()
            self.llm_enabled = tk.BooleanVar()
            self.barge_in = tk.BooleanVar()

            ttk.Checkbutton(
                toggles,
                text="Wake word",
                variable=self.wake_enabled,
                command=lambda: self._set_bool("wake_word.enabled", self.wake_enabled.get()),
            ).pack(anchor="w", padx=8, pady=4)
            ttk.Checkbutton(
                toggles,
                text="LLM (Ollama)",
                variable=self.llm_enabled,
                command=lambda: self._set_bool("llm.enabled", self.llm_enabled.get()),
            ).pack(anchor="w", padx=8, pady=4)
            ttk.Checkbutton(
                toggles,
                text="Barge-in (перебить голосом)",
                variable=self.barge_in,
                command=lambda: self._set_bool("audio.barge_in_enabled", self.barge_in.get()),
            ).pack(anchor="w", padx=8, pady=4)

            # Mic device
            mic = ttk.LabelFrame(root, text="Микрофон")
            mic.pack(fill=tk.X, pady=(12, 0))

            self.mic_choice = tk.StringVar()
            self._mic_map: dict[str, int | None] = {"Авто": None}
            try:
                devices = sd.query_devices()
            except Exception:
                devices = []
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) <= 0:
                    continue
                name = f"{idx}: {dev.get('name','?')}"
                self._mic_map[name] = idx
            mic_values = list(self._mic_map.keys())
            ttk.Combobox(mic, textvariable=self.mic_choice, values=mic_values, state="readonly").pack(
                fill=tk.X, padx=8, pady=8
            )
            ttk.Button(mic, text="Применить", command=self._apply_mic).pack(anchor="e", padx=8, pady=(0, 8))

            bottom = ttk.Frame(root)
            bottom.pack(fill=tk.X, pady=(12, 0))
            ttk.Button(bottom, text="Открыть config.yaml", command=self._open_config).pack(side=tk.RIGHT)
            ttk.Button(bottom, text="Reload", command=self._reload).pack(side=tk.RIGHT, padx=(0, 8))

            self.status_line = tk.StringVar(value="")
            ttk.Label(root, textvariable=self.status_line).pack(anchor="w", pady=(10, 0), padx=2)

        def _build_logs(self, root, tk, ttk) -> None:
            bar = ttk.Frame(root)
            bar.pack(fill=tk.X)
            ttk.Label(bar, text="jarvis.log").pack(side=tk.LEFT)
            self._log_hint = tk.StringVar(value="")
            ttk.Label(bar, textvariable=self._log_hint).pack(side=tk.RIGHT)

            frame = ttk.Frame(root)
            frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

            self._log_text = tk.Text(
                frame,
                wrap="none",
                height=20,
                bg="#151515",
                fg="#e6e6e6",
                insertbackground="#e6e6e6",
                relief="flat",
            )
            self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            y = ttk.Scrollbar(frame, orient="vertical", command=self._log_text.yview)
            y.pack(side=tk.RIGHT, fill=tk.Y)
            self._log_text.configure(yscrollcommand=y.set)
            self._log_text.configure(state="disabled")

        def _slider(self, parent: ttk.LabelFrame, label: str, var: tk.DoubleVar, lo: float, hi: float, key: str) -> None:
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, padx=8, pady=6)
            ttk.Label(row, text=label, width=26).pack(side=tk.LEFT)
            s = ttk.Scale(row, variable=var, from_=lo, to=hi, orient=tk.HORIZONTAL, command=lambda _v: self._debounced_set(key, var.get()))
            s.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))

        def _debounced_set(self, key: str, value: float) -> None:
            if hasattr(self, "_debounce"):
                try:
                    self.after_cancel(self._debounce)  # type: ignore[attr-defined]
                except Exception:
                    pass
            self._debounce = self.after(250, lambda: self._set_value(key, f"{value:.3f}"))  # type: ignore[attr-defined]

        def _reload(self) -> None:
            self.wake_enabled.set(bool(get_dotted(self._config_path, "wake_word.enabled", True)))
            self.llm_enabled.set(bool(get_dotted(self._config_path, "llm.enabled", True)))
            self.barge_in.set(bool(get_dotted(self._config_path, "audio.barge_in_enabled", True)))

            cur_dev = get_dotted(self._config_path, "audio.input_device", None)
            choice = "Авто"
            for k, v in self._mic_map.items():
                if v == cur_dev:
                    choice = k
                    break
            self.mic_choice.set(choice)

            self.status_line.set(f"Config: {self._config_path} | mode: {self._svc.mode}")

        def _poll(self) -> None:
            st = _read_status(self._status_path)
            state = str(st.get("state", "offline"))
            msg = str(st.get("message", ""))
            lvl = float(st.get("mic_level", 0.0) or 0.0)
            pct = int(max(0.0, min(1.0, lvl / 0.2)) * 100)

            self._st_state.set(state)
            self._st_msg.set(msg)
            self._mic_pct.set(f"{pct}%")
            try:
                self._mic_bar["value"] = pct
            except Exception:
                pass

            heard = str(st.get("last_heard", "") or "")
            err = str(st.get("last_error", "") or "")
            self._st_last.set(heard if heard else "—")
            self._st_err.set(err if err else "—")

            lr = float(st.get("latency_record_ms", 0.0) or 0.0)
            ls = float(st.get("latency_stt_ms", 0.0) or 0.0)
            ll = float(st.get("latency_llm_ms", 0.0) or 0.0)
            lt = float(st.get("latency_tts_ms", 0.0) or 0.0)
            ltot = float(st.get("latency_total_ms", 0.0) or 0.0)
            self._st_lat.set(f"lat(ms): rec={lr:.0f} stt={ls:.0f} llm={ll:.0f} tts={lt:.0f} total={ltot:.0f}")

            # logs
            lines = _tail_lines(self._log_path, max_lines=300)
            self._log_hint.set(str(self._log_path))
            if hasattr(self, "_log_text"):
                try:
                    self._log_text.configure(state="normal")
                    self._log_text.delete("1.0", tk.END)
                    self._log_text.insert(tk.END, "\n".join(lines) + ("\n" if lines else ""))
                    self._log_text.see(tk.END)
                    self._log_text.configure(state="disabled")
                except Exception:
                    pass

            self.after(500, self._poll)

        def _set_bool(self, key: str, value: bool) -> None:
            self._set_value(key, "true" if value else "false")

        def _set_value(self, key: str, value: str) -> None:
            try:
                set_dotted(self._config_path, key, value)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

        def _apply_preset(self) -> None:
            try:
                if self.preset.get() == "speed":
                    apply_speed_preset(self._config_path, template_path=self._template_path)
                else:
                    apply_room_preset(self._config_path, template_path=self._template_path)
                self._reload()
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

        def _apply_mic(self) -> None:
            v = self._mic_map.get(self.mic_choice.get(), None)
            self._set_value("audio.input_device", "null" if v is None else str(int(v)))

        def _start(self) -> None:
            if self._svc.mode == "systemd":
                _systemctl_user("start", "jarvis.service")
                return
            threading.Thread(target=_portable_start, args=(self._config_path, self._svc.pid_file), daemon=True).start()

        def _stop(self) -> None:
            if self._svc.mode == "systemd":
                _systemctl_user("stop", "jarvis.service")
                return
            _portable_stop(self._svc.pid_file)

        def _restart(self) -> None:
            if self._svc.mode == "systemd":
                _systemctl_user("restart", "jarvis.service")
                return
            self._stop()
            self._start()

        def _open_config(self) -> None:
            subprocess.Popen(["xdg-open", str(self._config_path)])

    return JarvisUI


def main(argv: list[str] | None = None) -> int:
    try:
        JarvisUI = _make_ui_class()
    except ModuleNotFoundError as e:
        if getattr(e, "name", None) == "tkinter":
            print(
                "GUI requires Tkinter. On Ubuntu/Debian install it with: sudo apt install python3-tk",
                file=sys.stderr,
            )
            return 2
        if getattr(e, "name", None) == "sounddevice":
            print("GUI requires 'sounddevice' package", file=sys.stderr)
            return 2
        raise
    except Exception as e:
        print(f"GUI cannot start: {e}", file=sys.stderr)
        return 2

    cfg = find_config_path(None)
    JarvisUI(config_path=cfg).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
