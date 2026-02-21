from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
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
