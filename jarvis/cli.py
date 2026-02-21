from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from jarvis.app import run
from jarvis.monitor import main as monitor_main
from jarvis.paths import ensure_default_config, find_config_path, get_paths
from jarvis.tray_app_system import main as tray_main
from jarvis.yaml_config import get_dotted, set_dotted
from jarvis.presets import apply_room_preset, apply_speed_preset


def _cmd_run(args: argparse.Namespace) -> int:
    run(str(find_config_path(args.config)))
    return 0


def _cmd_tray(_args: argparse.Namespace) -> int:
    tray_main()
    return 0


def _cmd_ui(_args: argparse.Namespace) -> int:
    from jarvis.ui_window import main as ui_main

    return int(ui_main())


def _cmd_monitor(args: argparse.Namespace) -> int:
    return int(monitor_main(["--status", str(args.status), "--interval", str(args.interval)]))


def _cmd_devices(_args: argparse.Namespace) -> int:
    from jarvis.mic_tools import list_devices

    devs = list_devices()
    # show only input-capable devices first
    inputs = [d for d in devs if d.max_input_channels > 0]
    others = [d for d in devs if d.max_input_channels <= 0]

    def fmt(d):
        return f"{d.index:>3} in={d.max_input_channels} out={d.max_output_channels} sr={int(d.default_samplerate):>5} {d.hostapi}: {d.name}"

    for d in inputs:
        print(fmt(d))
    if others:
        print("\n(no input)")
        for d in others[:10]:
            print(fmt(d))
    return 0


def _cmd_mic_test(args: argparse.Namespace) -> int:
    from jarvis.mic_tools import mic_test

    device = args.device
    if device is not None:
        try:
            device = int(device)
        except Exception:
            device = str(device)

    avg, peak, n = mic_test(seconds=float(args.seconds), sample_rate=int(args.sample_rate), device=device)
    print(f"chunks={n} avg_rms={avg:.4f} peak_rms={peak:.4f}")
    if peak < 0.002:
        print("Похоже, вход очень тихий/не тот микрофон. Попробуйте выбрать другой device: jarvis devices")
    elif peak < 0.006:
        print("Вход тихий. Для wake word попробуйте уменьшить пороги:")
        print("  jarvis config set wake_word.min_rms 0.001")
        print("  jarvis config set wake_word.noise_gate_multiplier 2.5")
    else:
        print("Микрофон выглядит живым.")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    paths = get_paths()
    template = (Path(__file__).resolve().parents[1] / "config.yaml").resolve()
    dest = paths.config_path if args.dest is None else Path(args.dest).expanduser().resolve()
    ensure_default_config(template_path=template, dest_path=dest)
    print(f"Config: {dest}")
    print(f"Data dir: {paths.data_dir}")
    print(f"State dir: {paths.state_dir}")
    return 0


def _cmd_config_get(args: argparse.Namespace) -> int:
    cfg_path = find_config_path(args.config)
    template = (Path(__file__).resolve().parents[1] / "config.yaml").resolve()
    ensure_default_config(template_path=template, dest_path=cfg_path)
    val = get_dotted(cfg_path, args.key, default=None)
    print(val)
    return 0


def _cmd_config_set(args: argparse.Namespace) -> int:
    cfg_path = find_config_path(args.config)
    template = (Path(__file__).resolve().parents[1] / "config.yaml").resolve()
    ensure_default_config(template_path=template, dest_path=cfg_path)
    set_dotted(cfg_path, args.key, args.value)
    print(f"OK: {args.key} = {args.value}")
    return 0


def _cmd_config_toggle(args: argparse.Namespace) -> int:
    cfg_path = find_config_path(args.config)
    template = (Path(__file__).resolve().parents[1] / "config.yaml").resolve()
    ensure_default_config(template_path=template, dest_path=cfg_path)
    cur = bool(get_dotted(cfg_path, args.key, default=False))
    set_dotted(cfg_path, args.key, (not cur))
    print(f"OK: {args.key} = {str(not cur).lower()}")
    return 0


def _cmd_config_open(args: argparse.Namespace) -> int:
    cfg_path = find_config_path(args.config)
    template = (Path(__file__).resolve().parents[1] / "config.yaml").resolve()
    ensure_default_config(template_path=template, dest_path=cfg_path)
    subprocess.Popen(["xdg-open", str(cfg_path)])
    return 0


def _systemctl_user(*args: str) -> int:
    try:
        r = subprocess.run(["systemctl", "--user", *args], check=False)
        return int(r.returncode)
    except Exception:
        return 1


def _cmd_service(args: argparse.Namespace) -> int:
    unit = args.unit
    action = args.action
    if action == "status":
        return _systemctl_user("status", "--no-pager", unit)
    return _systemctl_user(action, unit)


def _cmd_preset_speed(args: argparse.Namespace) -> int:
    cfg_path = find_config_path(args.config)
    template = (Path(__file__).resolve().parents[1] / "config.yaml").resolve()
    apply_speed_preset(cfg_path, template_path=template)

    print("OK: speed preset applied")
    print(f"Config: {cfg_path}")
    return 0


def _cmd_preset_room(args: argparse.Namespace) -> int:
    cfg_path = find_config_path(args.config)
    template = (Path(__file__).resolve().parents[1] / "config.yaml").resolve()
    apply_room_preset(cfg_path, template_path=template)

    print("OK: room preset applied")
    print(f"Config: {cfg_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jarvis", description="Локальный голосовой ассистент (RU)")
    p.add_argument("--config", default=None, help="Путь к config.yaml (legacy/top-level)")
    sub = p.add_subparsers(dest="cmd", required=False)

    p_run = sub.add_parser("run", help="Запустить ассистента")
    p_run.add_argument("--config", default=None, help="Путь к config.yaml (иначе XDG/portable)")
    p_run.set_defaults(func=_cmd_run)

    p_tray = sub.add_parser("tray", help="Запустить трей")
    p_tray.set_defaults(func=_cmd_tray)

    p_ui = sub.add_parser("ui", help="Окно настроек")
    p_ui.set_defaults(func=_cmd_ui)

    p_mon = sub.add_parser("monitor", help="Статус в терминале")
    default_status = get_paths().status_path
    p_mon.add_argument("--status", default=default_status, type=Path)
    p_mon.add_argument("--interval", default=0.2, type=float)
    p_mon.set_defaults(func=_cmd_monitor)

    p_dev = sub.add_parser("devices", help="Список аудио-устройств")
    p_dev.set_defaults(func=_cmd_devices)

    p_mt = sub.add_parser("mic-test", help="Проверка микрофона (RMS)")
    p_mt.add_argument("--device", default=None, help="Индекс устройства (см. jarvis devices)")
    p_mt.add_argument("--seconds", default=5.0, type=float)
    p_mt.add_argument("--sample-rate", default=16000, type=int)
    p_mt.set_defaults(func=_cmd_mic_test)

    p_init = sub.add_parser("init", help="Создать config.yaml в XDG")
    p_init.add_argument("--dest", default=None, help="Куда записать config.yaml (по умолчанию XDG)")
    p_init.set_defaults(func=_cmd_init)

    p_cfg = sub.add_parser("config", help="Управление конфигом без ручного редактирования")
    p_cfg.add_argument("--config", default=None, help="Явный путь к config.yaml")
    cfg_sub = p_cfg.add_subparsers(dest="cfg_cmd", required=True)

    p_get = cfg_sub.add_parser("get", help="Прочитать значение ключа")
    p_get.add_argument("key", help="Напр. llm.enabled")
    p_get.set_defaults(func=_cmd_config_get)

    p_set = cfg_sub.add_parser("set", help="Установить значение ключа")
    p_set.add_argument("key", help="Напр. llm.model")
    p_set.add_argument("value", help="Напр. true / 123 / qwen2.5:1.5b-instruct")
    p_set.set_defaults(func=_cmd_config_set)

    p_tog = cfg_sub.add_parser("toggle", help="Переключить булевый ключ")
    p_tog.add_argument("key", help="Напр. llm.enabled")
    p_tog.set_defaults(func=_cmd_config_toggle)

    p_open = cfg_sub.add_parser("open", help="Открыть config.yaml")
    p_open.set_defaults(func=_cmd_config_open)

    p_srv = sub.add_parser("service", help="Управление systemd --user сервисами")
    p_srv.add_argument("action", choices=["start", "stop", "restart", "status"], help="Действие")
    p_srv.add_argument(
        "unit",
        choices=["jarvis.service", "jarvis-tray.service", "ollama.service"],
        help="Юнит",
    )
    p_srv.set_defaults(func=_cmd_service)

    p_preset = sub.add_parser("preset", help="Готовые пресеты настроек")
    preset_sub = p_preset.add_subparsers(dest="preset_cmd", required=True)

    p_speed = preset_sub.add_parser("speed", help="Максимальная скорость реакции")
    p_speed.add_argument("--config", default=None, help="Явный путь к config.yaml")
    p_speed.set_defaults(func=_cmd_preset_speed)

    p_room = preset_sub.add_parser("room", help="Комната/дальний микрофон (меньше ложных срабатываний)")
    p_room.add_argument("--config", default=None, help="Явный путь к config.yaml")
    p_room.set_defaults(func=_cmd_preset_room)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    # backward-compat: `python -m jarvis --config ...`
    if args.cmd is None:
        # emulate legacy: treat as `run` with optional --config
        run(str(find_config_path(args.config)))
        return 0

    func = getattr(args, "func", None)
    if func is None:
        p.print_help()
        return 2
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
