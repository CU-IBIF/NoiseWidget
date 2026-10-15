"""Headless runner: ``python -m noisewidget`` prints the explained noise budget.

The same model the widget draws, with the same numbers, for a terminal or a test.
Any lever can be set with ``--set name=value``; ``--ui`` opens the window.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from .physics.acquisition import Levers, Simulator
from .physics.confocal import ScanLevers, ScanSimulator
from .physics.presets import CONFOCAL_PRESETS, PRESETS
from .physics.quantitation import QuantSettings, quantify
from .physics.report import build_report, format_report
from .physics.scan_report import build_scan_report

# frame -> (default levers, simulator, report, presets)
FRAMES = {
    "camera": (Levers, Simulator, build_report, PRESETS),
    "confocal": (ScanLevers, ScanSimulator, build_scan_report, CONFOCAL_PRESETS),
}


def parse_assignment(levers, text: str):
    """Apply one ``name=value`` to the levers (either frame's), coercing types."""
    if "=" not in text:
        raise SystemExit(f"--set expects name=value, got {text!r}")
    name, raw = (part.strip() for part in text.split("=", 1))
    noise_names = {f.name for f in dataclasses.fields(type(levers.noise))}
    if name in noise_names:
        on = raw.lower() in ("1", "true", "on", "yes")
        return levers.with_(noise=dataclasses.replace(levers.noise, **{name: on}))
    current = {f.name: f for f in dataclasses.fields(type(levers))}
    if name not in current or name == "noise":
        known = ", ".join(sorted(set(current) - {"noise"} | noise_names))
        raise SystemExit(f"unknown lever {name!r}; known: {known}")
    value = getattr(levers, name)
    if raw.lower() == "none":
        coerced = None
    elif isinstance(value, bool):
        coerced = raw.lower() in ("1", "true", "on", "yes")
    elif isinstance(value, int):
        coerced = int(raw)
    elif isinstance(value, float) or value is None:
        try:
            coerced = float(raw)
        except ValueError:
            coerced = raw
    else:
        coerced = raw
    return levers.with_(**{name: coerced})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="noisewidget")
    parser.add_argument("--ui", action="store_true", help="open the widget")
    parser.add_argument(
        "--frame", choices=sorted(FRAMES), default="camera",
        help="camera (widefield) or confocal (point scanning); presets choose their own",
    )
    parser.add_argument("--preset", help="start from a named preset (see --list-presets)")
    parser.add_argument("--list-presets", action="store_true", help="list the presets")
    parser.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE",
        help="set a lever, e.g. --set na=0.75 --set medium=air --set read=off",
    )
    parser.add_argument("--realizations", type=int, default=100, help="noise realizations for quantitation")
    args = parser.parse_args(argv)

    if args.list_presets:
        for frame, (_, _, _, presets) in FRAMES.items():
            print(f"[{frame}]")
            for name, preset in presets.items():
                print(f"  {name}\n      {preset.description}")
        return 0

    frame = args.frame
    if args.preset:
        owner = [f for f, (_, _, _, presets) in FRAMES.items() if args.preset in presets]
        if not owner:
            print(f"no preset {args.preset!r}; try --list-presets")
            return 2
        frame = owner[0]
    make_levers, make_sim, report, presets = FRAMES[frame]
    levers = presets[args.preset].levers if args.preset else make_levers()
    for item in args.set:
        levers = parse_assignment(levers, item)

    if args.ui:
        try:
            from .ui.widget import launch
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            print(f"the widget needs the 'ui' extra: pip install 'noisewidget[ui]' ({exc})")
            return 2
        return launch(levers, frame=frame)

    try:
        sim = make_sim().run(levers)
    except ValueError as exc:
        print(f"cannot image this: {exc}")
        return 2
    q = quantify(sim, QuantSettings(realizations=args.realizations))
    instrument = levers.camera if frame == "camera" else f"{levers.detector} ({levers.acquisition_mode})"
    print(f"[{frame}]  {levers.specimen}  |  {levers.fluorophore}  |  {levers.magnification:g}x / NA "
          f"{levers.na:g} {levers.medium}  |  {instrument}")
    print(format_report(report(sim, q)))
    for note in sim.notes:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
