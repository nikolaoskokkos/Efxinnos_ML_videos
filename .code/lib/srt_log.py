"""Reads and writes the .srt telemetry log that comes with each dive video (used by steps 0, 5 and 6).

Each entry looks like:
    12
    00:00:05,500 --> 00:00:06,000
    2026/07/16-10:48:05 -- lat:40.828141 lon:24.340483 imm.:4.012220 alt.:12.429604 -- light state:1
"""
import re
from datetime import datetime

import numpy as np


def _seconds(stamp):
    h, m, s = stamp.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _stamp(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _number(key, text):
    match = re.search(rf"{key}\s*:\s*(-?[\d.]+)", text)
    return float(match.group(1)) if match else np.nan


def read_srt(path):
    """List of entries, each a dict: start, end (seconds of video), text (the data lines),
    clock (datetime or None), lat, lon, depth (imm.), alt (alt.), light.

    Some logs label their entries 0.5 s apart while writing them faster: GX010175 has about
    6 entries per second, so its subtitle times run 3x too slow (14:53 of subtitles for a
    4:50 video). When the subtitle times and the logged clock disagree by more than 10 %,
    every entry is timed by its clock instead (entries within one clock second are spread
    evenly over it), and `retimed` is set on the list's entries."""
    entries = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace").strip()):
        lines = block.strip().splitlines()
        timing = next((l for l in lines if "-->" in l), None)
        if timing is None:
            continue
        start, end = (_seconds(t.strip()) for t in timing.split("-->"))
        data = [l for l in lines if "-->" not in l and not l.strip().isdigit()]
        text = " ".join(data)
        when = re.search(r"\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}", text)
        entries.append({
            "start": start, "end": end, "text": data,
            "clock": datetime.strptime(when.group(0), "%Y/%m/%d-%H:%M:%S") if when else None,
            "lat": _number("lat", text), "lon": _number("lon", text),
            "depth": _number(r"imm\.", text), "alt": _number(r"alt\.", text),
            "light": _number("light state", text), "retimed": False,
        })
    _retime(entries)
    return entries


def _retime(entries):
    if len(entries) < 3 or any(e["clock"] is None for e in entries):
        return
    clock = np.array([(e["clock"] - entries[0]["clock"]).total_seconds() for e in entries])
    if np.any(np.diff(clock) < 0):
        return  # clock jumps back: can't be trusted
    clock_span = clock[-1] + 1  # the last clock second is a whole second long
    label_span = entries[-1]["end"] - entries[0]["start"]
    if abs(label_span / clock_span - 1) <= 0.1:
        return
    times = clock.astype(float)
    for second in np.unique(clock):
        same = np.flatnonzero(clock == second)
        times[same] = second + np.arange(len(same)) / len(same)
    step = 1 / np.sum(clock == clock[-1])
    for i, e in enumerate(entries):
        e["start"] = float(times[i])
        e["end"] = float(times[i + 1]) if i + 1 < len(entries) else float(times[i] + step)
        e["retimed"] = True


def describe(entries, path):
    """One line for the console: number of entries, and a note when the log was retimed."""
    note = ""
    if entries and entries[0]["retimed"]:
        rate = len(entries) / (entries[-1]["end"] - entries[0]["start"])
        note = f"; its times were wrong (labelled 0.5 s apart, really ~{rate:.1f} per second): timed by its clock"
    return f"{path}: {len(entries)} entries, {entries[-1]['end'] if entries else 0:.0f} s{note}"


def write_srt(entries, path, offset=0.0, until=None):
    """Write entries with their (corrected) times, shifted by -offset and cut at `until`."""
    blocks = []
    for e in entries:
        a, b = e["start"] - offset, e["end"] - offset
        if b <= 0 or (until is not None and a >= until):
            continue
        b = b if until is None else min(b, until)
        blocks.append("\n".join([str(len(blocks) + 1), f"{_stamp(max(a, 0))} --> {_stamp(b)}", *e["text"]]))
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
