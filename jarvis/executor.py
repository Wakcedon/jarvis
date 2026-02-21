from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from jarvis.capabilities import Capabilities
from jarvis.desktop_apps import find_app, launch_app


def _resolve_exe(name: str) -> str | None:
    return which(name)


def _run(cmd: list[str], *, timeout_s: float = 6.0) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_s)
        return True
    except Exception:
        return False


@dataclass
class SystemExecutor:
    caps: Capabilities

    def open_url(self, url: str) -> bool:
        if not self.caps.can_open_urls:
            return False
        try:
            subprocess.Popen(["xdg-open", url])
            return True
        except Exception:
            return False

    def open_path(self, path: Path) -> bool:
        if not self.caps.can_open_paths:
            return False
        try:
            subprocess.Popen(["xdg-open", str(path)])
            return True
        except Exception:
            return False

    def launch_app(self, exe: str) -> bool:
        if not self.caps.can_launch_apps:
            return False
        try:
            subprocess.Popen([exe])
            return True
        except Exception:
            return False

    def launch_desktop(self, query: str) -> bool:
        if not self.caps.can_desktop_launch:
            return False
        app = find_app(query)
        if not app:
            return False
        return bool(launch_app(app))

    def run_shell(self, cmd: str) -> bool:
        if not self.caps.can_shell:
            return False
        try:
            subprocess.Popen(cmd, shell=True)
            return True
        except Exception:
            return False

    # --- Mic ---
    def mic_mute(self, mute: bool) -> bool:
        if not self.caps.can_mic_mute:
            return False
        wpctl = _resolve_exe("wpctl")
        if wpctl:
            return _run([wpctl, "set-mute", "@DEFAULT_AUDIO_SOURCE@", "1" if mute else "0"])
        pactl = _resolve_exe("pactl")
        if pactl:
            return _run([pactl, "set-source-mute", "@DEFAULT_SOURCE@", "1" if mute else "0"])
        return False

    # --- Network ---
    def set_radio(self, kind: str, on: bool) -> bool:
        if not self.caps.can_network:
            return False
        nmcli = _resolve_exe("nmcli")
        if nmcli:
            return _run([nmcli, "radio", kind, "on" if on else "off"], timeout_s=6.0)
        if kind == "wifi":
            rfkill = _resolve_exe("rfkill")
            if rfkill:
                return _run([rfkill, "unblock" if on else "block", "wifi"], timeout_s=6.0)
        if kind == "bluetooth":
            bluetoothctl = _resolve_exe("bluetoothctl")
            if bluetoothctl:
                return _run([bluetoothctl, "power", "on" if on else "off"], timeout_s=6.0)
        return False

    # --- Window / Media ---
    def xdotool_key(self, keys: str) -> bool:
        if not self.caps.can_window:
            return False
        xdotool = _resolve_exe("xdotool")
        if not xdotool:
            return False
        return _run([xdotool, "key", keys], timeout_s=2.0)

    def playerctl(self, cmd: str) -> bool:
        if not self.caps.can_media:
            return False
        playerctl = _resolve_exe("playerctl")
        if not playerctl:
            return False
        return _run([playerctl, cmd], timeout_s=2.0)

    # --- Clipboard ---
    def clipboard_set(self, text: str) -> bool:
        if not self.caps.can_clipboard:
            return False
        wl = _resolve_exe("wl-copy")
        if wl:
            try:
                subprocess.run([wl], input=text.encode("utf-8"), check=True, timeout=2.0)
                return True
            except Exception:
                return False
        xclip = _resolve_exe("xclip")
        if xclip:
            try:
                subprocess.run([xclip, "-selection", "clipboard"], input=text.encode("utf-8"), check=True, timeout=2.0)
                return True
            except Exception:
                return False
        return False

    def clipboard_get(self) -> str | None:
        if not self.caps.can_clipboard:
            return None
        wl = _resolve_exe("wl-paste")
        if wl:
            try:
                out = subprocess.check_output([wl, "-n"], timeout=2.0)
                s = out.decode("utf-8", errors="ignore").strip()
                return s or None
            except Exception:
                return None
        xclip = _resolve_exe("xclip")
        if xclip:
            try:
                out = subprocess.check_output([xclip, "-selection", "clipboard", "-o"], timeout=2.0)
                s = out.decode("utf-8", errors="ignore").strip()
                return s or None
            except Exception:
                return None
        return None

    # --- Volume ---
    def volume_step(self, delta_percent: int) -> bool:
        if not self.caps.can_volume:
            return False
        wpctl = _resolve_exe("wpctl")
        if wpctl:
            sign = "+" if delta_percent > 0 else "-"
            return _run([wpctl, "set-volume", "@DEFAULT_AUDIO_SINK@", f"{abs(delta_percent)}%{sign}"])
        pactl = _resolve_exe("pactl")
        if pactl:
            op = "+" if delta_percent > 0 else "-"
            return _run([pactl, "set-sink-volume", "@DEFAULT_SINK@", f"{abs(delta_percent)}%{op}"])
        return False

    def volume_set_percent(self, percent: int) -> bool:
        if not self.caps.can_volume:
            return False
        wpctl = _resolve_exe("wpctl")
        if wpctl:
            return _run([wpctl, "set-volume", "@DEFAULT_AUDIO_SINK@", f"{percent}%"])
        pactl = _resolve_exe("pactl")
        if pactl:
            return _run([pactl, "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"])
        return False

    def volume_mute(self, mute: bool) -> bool:
        if not self.caps.can_volume:
            return False
        wpctl = _resolve_exe("wpctl")
        if wpctl:
            return _run([wpctl, "set-mute", "@DEFAULT_AUDIO_SINK@", "1" if mute else "0"])
        pactl = _resolve_exe("pactl")
        if pactl:
            return _run([pactl, "set-sink-mute", "@DEFAULT_SINK@", "1" if mute else "0"])
        return False

    # --- Screenshots ---
    def take_screenshot(self, dir_path: Path, *, mode: str) -> Path | None:
        if not self.caps.can_screenshots:
            return None
        import datetime as dt

        dir_path.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = dir_path / f"screenshot_{ts}.png"
        gnome = _resolve_exe("gnome-screenshot")
        if not gnome:
            return None
        if mode == "window":
            ok = _run([gnome, "-w", "-f", str(out)], timeout_s=15.0)
        else:
            ok = _run([gnome, "-f", str(out)], timeout_s=15.0)
        return out if ok else None
