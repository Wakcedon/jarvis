from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jarvis.config import load_config
from jarvis.yaml_config import set_dotted_bool


def _try_import_indicator():
    import gi  # type: ignore

    gi.require_version("Gtk", "3.0")

    try:
        gi.require_version("AyatanaAppIndicator3", "0.1")
        from gi.repository import AyatanaAppIndicator3 as AppIndicator3  # type: ignore
    except Exception:
        gi.require_version("AppIndicator3", "0.1")
        from gi.repository import AppIndicator3  # type: ignore

    from gi.repository import GLib, Gtk  # type: ignore

    return AppIndicator3, GLib, Gtk


@dataclass(frozen=True)
class TrayState:
    status_text: str
    mic_level: float


def _read_status(path: Path) -> TrayState:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = str(data.get("state", "unknown"))
        msg = str(data.get("message", ""))
        lvl = float(data.get("mic_level", 0.0) or 0.0)
        status = f"{state}: {msg}" if msg else state
        return TrayState(status_text=status, mic_level=lvl)
    except Exception:
        return TrayState(status_text="offline", mic_level=0.0)


def _systemctl_user(*args: str) -> None:
    subprocess.run(["systemctl", "--user", *args], check=False)


def _edit_config_bool(config_path: Path, dotted_key: str, value: bool) -> None:
    set_dotted_bool(config_path, dotted_key, value)


def main() -> None:
    AppIndicator3, GLib, Gtk = _try_import_indicator()

    cfg = load_config("config.yaml")
    config_path = Path("config.yaml").resolve()
    status_path = cfg.ui.status_path

    icon_path = (Path(__file__).resolve().parents[1] / "assets" / "jarvis.svg")

    # AppIndicator чаще ожидает icon-name из темы, а не абсолютный путь.
    # Добавляем папку assets в поиск темы и используем имя "jarvis".
    try:
        Gtk.IconTheme.get_default().prepend_search_path(str(icon_path.parent))
    except Exception:
        pass

    indicator = AppIndicator3.Indicator.new(
        "jarvis",
        "jarvis",
        AppIndicator3.IndicatorCategory.APPLICATION_STATUS,
    )
    indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
    try:
        indicator.set_icon_full(str(icon_path), "Jarvis")
    except Exception:
        pass

    menu = Gtk.Menu()

    item_status = Gtk.MenuItem(label="Статус: …")
    item_status.set_sensitive(False)
    menu.append(item_status)

    item_mic = Gtk.MenuItem(label="Микрофон: …")
    item_mic.set_sensitive(False)
    menu.append(item_mic)

    menu.append(Gtk.SeparatorMenuItem())

    toggle_wake = Gtk.CheckMenuItem(label="Wake word (активация)")
    toggle_wake.set_active(cfg.wake_word.enabled)
    menu.append(toggle_wake)

    toggle_llm = Gtk.CheckMenuItem(label="LLM (Ollama)")
    toggle_llm.set_active(cfg.llm.enabled)
    menu.append(toggle_llm)

    menu.append(Gtk.SeparatorMenuItem())

    btn_restart = Gtk.MenuItem(label="Перезапустить Jarvis")
    menu.append(btn_restart)

    btn_stop = Gtk.MenuItem(label="Остановить Jarvis")
    menu.append(btn_stop)

    btn_start = Gtk.MenuItem(label="Запустить Jarvis")
    menu.append(btn_start)

    menu.append(Gtk.SeparatorMenuItem())

    btn_open_config = Gtk.MenuItem(label="Открыть config.yaml")
    menu.append(btn_open_config)

    btn_quit = Gtk.MenuItem(label="Выход")
    menu.append(btn_quit)

    menu.show_all()
    indicator.set_menu(menu)

    def on_toggle_wake(_item):
        _edit_config_bool(config_path, "wake_word.enabled", bool(toggle_wake.get_active()))
        _systemctl_user("restart", "jarvis.service")

    def on_toggle_llm(_item):
        _edit_config_bool(config_path, "llm.enabled", bool(toggle_llm.get_active()))
        _systemctl_user("restart", "jarvis.service")

    def on_restart(_item):
        _systemctl_user("restart", "jarvis.service")

    def on_stop(_item):
        _systemctl_user("stop", "jarvis.service")

    def on_start(_item):
        _systemctl_user("start", "jarvis.service")

    def on_open_config(_item):
        subprocess.Popen(["xdg-open", str(config_path)])

    def on_quit(_item):
        Gtk.main_quit()

    toggle_wake.connect("toggled", on_toggle_wake)
    toggle_llm.connect("toggled", on_toggle_llm)
    btn_restart.connect("activate", on_restart)
    btn_stop.connect("activate", on_stop)
    btn_start.connect("activate", on_start)
    btn_open_config.connect("activate", on_open_config)
    btn_quit.connect("activate", on_quit)

    def poll() -> bool:
        st = _read_status(status_path)
        pct = int(max(0.0, min(1.0, st.mic_level / 0.2)) * 100)
        item_status.set_label(f"Статус: {st.status_text}")
        item_mic.set_label(f"Микрофон: {pct}%")
        try:
            indicator.set_title(f"Jarvis — {st.status_text} — mic {pct}%")
        except Exception:
            pass
        return True

    GLib.timeout_add(500, poll)
    Gtk.main()


if __name__ == "__main__":
    main()
