"""
Regenerate every figure in ``figures/`` from nothing but the data.

The figures were made in two environments, see the README, so this script
makes the half of them that belongs to the environment it is run in:

.. code-block:: bash

    .venv/Scripts/python scripts/regenerate_figures.py simulation
    .venv-level-4/Scripts/python scripts/regenerate_figures.py level-4

Neither command clears a cache. To regenerate from scratch rather than from
the cached intermediate results, delete ``~/.solar-research-2026/cache``
first, and point ``ESIS_CACHE_DIR`` at an empty directory.

Several figures were made with arguments other than the defaults, which the
names of their files record, so each call is written out here in full.
"""

import os
import pathlib
import sys
import time
import traceback

import astropy.units as u
import joblib

# The `esis` of the simulation environment always caches in `~/.esis/cache`,
# where it would share results with every other `esis` on the machine and
# throw away theirs whenever the two disagree. Later versions of `esis` read
# `ESIS_CACHE_DIR`, so the same variable is honored here for the older one.
_esis_cache_default = pathlib.Path.home() / ".esis/cache"


class _Memory(joblib.Memory):
    def __init__(self, location=None, *args, **kwargs):
        if location is not None and pathlib.Path(location) == _esis_cache_default:
            location = os.environ.get("ESIS_CACHE_DIR", location)
        super().__init__(location, *args, **kwargs)


joblib.Memory = _Memory

# `iris` gives its query of the Heliophysics Events Knowledgebase five seconds,
# which the knowledgebase no longer reliably answers in.
import requests  # noqa: E402

_get = requests.get


def _get_patiently(*args, **kwargs):
    kwargs["timeout"] = max(kwargs.get("timeout") or 0, 120)
    return _get(*args, **kwargs)


requests.get = _get_patiently

import spd2026  # noqa: E402
from spd2026.figures._level_4 import (  # noqa: E402
    center_event_default,
    position_event_south,
)

# The figures of the second event are given names of their own, since the
# functions would otherwise write them over the figures of event E.
_south = spd2026.figures.default_path

figures = {
    "simulation": [
        ("iris_ee", {}, ["iris-ee-1.svg", "iris-ee-2.svg", "iris-ee-3.svg"]),
        ("iris_ee_gallery", {}, ["iris-ee-gallery.svg"]),
        ("blink", {}, ["blink.mp4"]),
        ("blink_channels", {}, ["blink-channels.mp4"]),
        ("mart_scene", {}, ["mart-scene.mp4"]),
        ("mart_spectra", {}, ["mart-spectra.mp4"]),
        ("mart_moments", {}, ["mart-moments.mp4"]),
        ("mart_moments", {"index_iteration": -1}, ["mart-moments-50.svg"]),
        ("cinemagraph", {"suffix": ".mp4"}, ["cinemagraph-channel-3.mp4"]),
        ("cinemagraph_channels", {}, ["cinemagraph-channels.mp4"]),
    ],
    "level-4": [
        ("level_4", {"center_box": None}, ["level-4-o-v.mp4"]),
        ("level_4", {}, ["level-4-o-v-box.mp4"]),
        (
            "level_4",
            {"center": center_event_default, "center_box": None},
            ["level-4-o-v-event.mp4"],
        ),
        ("level_4_velocity", {}, ["level-4-o-v-velocity.mp4"]),
        (
            "level_4_velocity",
            {"center": center_event_default},
            ["level-4-o-v-velocity-event.mp4"],
        ),
        ("level_4_lines", {}, ["level-4-lines.mp4"]),
        ("level_4_event", {}, ["level-4-event.mp4"]),
        (
            "level_4",
            {
                "center_box": position_event_south,
                "path": _south / "level-4-o-v-box-south.mp4",
            },
            ["level-4-o-v-box-south.mp4"],
        ),
        (
            "level_4",
            {
                "center": position_event_south,
                "center_box": None,
                "path": _south / "level-4-o-v-event-south.mp4",
            },
            ["level-4-o-v-event-south.mp4"],
        ),
        (
            "level_4_event",
            {
                "center": position_event_south,
                "velocity_limit": 40 * u.km / u.s,
                "path": _south / "level-4-event-south.mp4",
            },
            ["level-4-event-south.mp4"],
        ),
        ("level_4_event_history", {}, ["level-4-event-history.svg"]),
        ("level_4_event_history", {"animated": True}, ["level-4-event-history.mp4"]),
        (
            "level_4_event_history",
            {"curves": False, "marks": False},
            ["level-4-event-history-images.svg"],
        ),
        (
            "level_4_event_history",
            {"curves": False, "marks": False, "animated": True},
            ["level-4-event-history-images.mp4"],
        ),
        ("level_4_event_motion", {}, ["level-4-event-motion.svg"]),
        ("level_4_event_motion", {"animated": True}, ["level-4-event-motion.mp4"]),
    ],
}
"""
Each figure, as the name of its function, its arguments, and the files it
writes, grouped by the environment it was made in.
"""


def main(group: str) -> int:
    failures = 0
    for name, kwargs, expected in figures[group]:
        start = time.time()
        try:
            result = getattr(spd2026.figures, name)(**kwargs)
        except Exception:
            failures += 1
            print(f"FAILED {name}\n{traceback.format_exc()}", flush=True)
            continue
        paths = result if isinstance(result, list) else [result]
        names = [path.name for path in paths]
        if names != expected:
            failures += 1
            print(f"FAILED {name} wrote {names}, not {expected}", flush=True)
            continue
        print(f"{', '.join(names)} in {time.time() - start:.0f} s", flush=True)
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
