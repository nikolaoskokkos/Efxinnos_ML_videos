"""Helpers for posidonia_pipeline.ipynb: settings panels, running the steps with live output and
showing the results. The work itself is done by the scripts (0.enhance_video.py, detect_video.py,
6.make_coverage_map.py), which read config.ini; this module writes config.ini from
config_template.ini and the values chosen in the panels."""
import configparser
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import ipywidgets as w
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Image, Video, display

# The code sits in the hidden .code/ folder; HERE is the project folder above it
CODE = Path(__file__).resolve().parent
HERE = CODE.parent if CODE.name == ".code" else CODE
INPUT = HERE / "input"
RESULTS = HERE / "results"
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi")
STATE = {"video": None, "enhanced": None, "detected": None, "coverage_name": None, "model": None, "run": None}
STYLE = {"description_width": "190px"}
WIDE = w.Layout(width="560px")
RUN_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_")  # results/2026-11-08_101532_GX010175_sample


# ------------------------------------------------------------------
# run folders: results/<date>_<time>_<video>/ with enhancement/, prediction/, coverage/ and
# settings.ini (the settings used). Step 1 starts a new run; steps 2 and 3 add to it.
# ------------------------------------------------------------------

def new_run(video):
    run = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M%S}_{Path(video).stem}"
    run.mkdir(parents=True, exist_ok=True)
    STATE["run"] = run
    print(f"Results of this run: {run.relative_to(HERE).as_posix()}/")
    return run


def current_run(video):
    """The run step 1 started, or a new one (e.g. step 2 run first after a kernel restart)."""
    run = STATE["run"]
    return run if run is not None and run.exists() else new_run(video)


def _save_settings(run):
    shutil.copy(HERE / "config.ini", run / "settings.ini")


def _rel(path):
    return Path(path).relative_to(HERE).as_posix()


def runs():
    """Run folders, newest first."""
    found = [d for d in RESULTS.iterdir() if d.is_dir() and RUN_NAME.match(d.name)] if RESULTS.exists() else []
    return sorted(found, key=lambda d: d.name, reverse=True)


def _find(sub, filename, run=None):
    """results/<run>/<sub>/<filename>: in the given run, the current one, or else the newest run
    that has it (also results/<sub>/ from before the runs had their own folders)."""
    places = [Path(run) if run and Path(run).is_absolute() else RESULTS / run] if run else []
    places += ([STATE["run"]] if STATE["run"] is not None else []) + runs() + [RESULTS]
    for place in places:
        if (place / sub / filename).exists():
            return place / sub / filename
    raise FileNotFoundError(f"{sub}/{filename} is not in any run folder in results/")


# ------------------------------------------------------------------
# environment and input
# ------------------------------------------------------------------

def check_environment():
    import torch
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    try:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
    except Exception:
        gpu = "none (CPU only)"
    print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, OpenCV {cv2.__version__}")
    print(f"CPU cores: {os.cpu_count()}, GPU: {gpu}")
    print(f"ffmpeg: {ffmpeg or 'not found - videos are read / written with OpenCV (slower)'}")
    videos = [p.name for p in input_videos()]
    print(f"Videos in input/: {', '.join(videos) if videos else 'none'}")


def input_videos():
    """Original videos in input/ (not the enhanced ones the notebook made)."""
    return sorted(p for p in INPUT.iterdir() if p.suffix.lower() in VIDEO_EXT and "_enhanced_" not in p.stem)


def download(url, name=None):
    """Download a file (video or .srt) into input/."""
    name = name or Path(url.split("?")[0]).name
    target = INPUT / name
    print(f"Downloading {url} -> {target} ...")
    urllib.request.urlretrieve(url, target)
    print(f"Done: {target.stat().st_size / 1e6:.1f} MB")
    return target


def _last_video():
    """Name of the video in config.ini (the last one used), else the template's (GX010287_sample.mp4)."""
    cfg = configparser.ConfigParser()
    target = HERE / "config.ini"
    cfg.read(target if target.exists() else HERE / "config_template.ini", encoding="utf-8")
    return Path(cfg.get("enhance", "video", fallback="")).name


def choose_video():
    """Panel: pick a video from input/ (with its .srt next to it), upload or download one.
    It starts on the video used last time (config.ini)."""
    pick = w.Dropdown(description="Video", style=STYLE, layout=WIDE)
    info = w.HTML()

    def refresh(*_):
        names = [p.name for p in input_videos()]
        current = pick.value or _last_video()
        pick.options = names
        if names:
            pick.value = current if current in names else names[0]

    def changed(*_):
        if not pick.value:
            info.value = "<i>No video in input/ yet.</i>"
            return
        video = INPUT / pick.value
        STATE["video"] = video
        cap = cv2.VideoCapture(str(video))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        cap.release()
        srt = video.with_suffix(".srt")
        srt_note = (f"<span style='color:green'>{srt.name} found</span>" if srt.exists() else
                    "<span style='color:#c00'>no .srt with the same name: enhancement uses a fixed "
                    "altitude, the coverage map needs the .srt</span>")
        info.value = f"{size} px, {n / fps:.0f} s at {fps:.2f} fps, {video.stat().st_size / 1e6:.0f} MB &nbsp; {srt_note}"

    upload = w.FileUpload(accept=".mp4,.MP4,.mov,.srt", multiple=True, description="Upload")
    url = w.Text(placeholder="https://... direct link to a .mp4 or .srt", description="or download URL",
                 style=STYLE, layout=WIDE)
    get = w.Button(description="Download", icon="download")
    log = w.Output()

    def uploaded(change):
        with log:
            for f in upload.value:
                (INPUT / f["name"]).write_bytes(bytes(f["content"]))
                print(f"Saved input/{f['name']} ({f['size'] / 1e6:.1f} MB)")
        refresh()

    def fetch(_):
        with log:
            try:
                download(url.value.strip())
            except Exception as e:
                print(f"Download failed: {e}")
        refresh()

    again = w.Button(description="Refresh", icon="refresh", layout=w.Layout(width="110px"))
    upload.observe(uploaded, names="value")
    get.on_click(fetch)
    again.on_click(refresh)
    pick.observe(changed, names="value")
    refresh()
    changed()
    display(w.VBox([pick, info, w.HTML("<b>Add a video</b> (with its .srt, same name). For big files use the "
                                       "file browser on the left: open <code>input/</code> and drag the files there, "
                                       "then press Refresh."),
                    w.HBox([upload, again]), w.HBox([url, get]), log]))
    return pick


# ------------------------------------------------------------------
# settings panels
# ------------------------------------------------------------------

def _f(value, desc, lo, hi, step, tip=""):
    return w.FloatSlider(value=value, min=lo, max=hi, step=step, description=desc, style=STYLE,
                         layout=WIDE, tooltip=tip, readout_format=".2f")


def enhance_settings():
    """Panel for step 1 (natural method)."""
    s = {
        "run": w.Checkbox(value=True, description="Enhance the video (untick to detect on the original)",
                          style=STYLE, layout=WIDE),
        "start": w.Text(value="", description="Start (s or m:ss)", placeholder="empty = beginning", style=STYLE),
        "end": w.Text(value="", description="End (s or m:ss)", placeholder="empty = end", style=STYLE),
        "range": w.Dropdown(options=[("flat seabed (altitude from the .srt)", "srt"),
                                     ("Depth Anything V2 (per pixel, slower)", "depth_anything")],
                            value="srt", description="Distance to the seabed", style=STYLE, layout=WIDE),
        "strength": _f(1.0, "Strength", 0.0, 1.0, 0.05),
        "haze": _f(1.0, "Haze removal", 0.5, 1.5, 0.05),
        "far_colour_limit": _f(1.0, "Far colour limit", 1.0, 2.0, 0.05,
                               "Caps the colour change with distance (top of the frame); 1 = off"),
        "range_balance": _f(1.0, "Distance colour balance", 0.0, 1.0, 0.1,
                            "Gives far sand / rock the colour and detail of the near seabed (no yellow / brown top); 0 = off"),
        "local_contrast": _f(0.0, "Local contrast", 0.0, 4.0, 0.5,
                             "Brings out texture (CLAHE on brightness); 0 = off, 2 = clearly sharper"),
        "sand_brightness": w.IntSlider(value=80, min=60, max=95, description="Sand brightness (L)", style=STYLE,
                                       layout=WIDE),
        "shadow_balance": w.Checkbox(value=True, description="Shadow balance (no pink shadows)", style=STYLE,
                                     layout=WIDE),
        "depth_correction": w.Checkbox(value=True, description="Correct for camera depth (imm. in .srt)",
                                       style=STYLE, layout=WIDE),
        "remove_particles": w.Checkbox(value=False, description="Remove floating particles (orange streaks)",
                                       style=STYLE, layout=WIDE),
        "particle_method": w.Dropdown(options=[("median: middle value of 5 frames (most removed, slightly softer)", "median"),
                                               ("bright: only what is brighter than in 5 frames (sharper, more left)", "bright"),
                                               ("blur: fill with the colour around (fast, can leave smudges)", "blur")],
                                      value="median", description="Removal method", style=STYLE,
                                      layout=w.Layout(width=WIDE.width, display="none")),  # own layout: it gets hidden
        "output_max": w.Dropdown(options=[("1920 px", 1920), ("1280 px", 1280), ("960 px", 960)], value=1920,
                                 description="Saved video size", style=STYLE),
        "calib_frames": w.IntSlider(value=60, min=20, max=120, step=10, description="Frames to fit the water",
                                    style=STYLE, layout=WIDE),
        "preview_only": w.Checkbox(value=False, description="Preview only (fit + before/after picture, fast)",
                                   style=STYLE, layout=WIDE),
    }

    def show_method(*_):  # the removal method only while "Remove floating particles" is ticked
        s["particle_method"].layout.display = None if s["remove_particles"].value else "none"

    s["remove_particles"].observe(show_method, names="value")
    show_method()
    display(w.VBox([w.HTML("<b>Step 1: natural enhancement</b> &nbsp; <i>haze above 1 = clearer, but the darkest "
                           "seagrass can turn grey-purple; strength 0.75 suits very green, shallow videos</i>"),
                    *s.values()]))
    return s


def model_versions():
    """Trained models in models/ (v8, v10, ...), oldest first."""
    found = [d.name for d in (HERE / "models").glob("v*") if d.name[1:].isdigit() and (d / "model.pt").exists()]
    return sorted(found, key=lambda v: int(v[1:]))


def _model_label(version, latest=False):
    """'v12: posidonia, rock, debris (mask mAP50 0.610)' from models/<version>/info.txt,
    'latest (v13): ...' for the newest model. The mAP of models trained on different frames is measured
    on different validation frames."""
    info = HERE / "models" / version / "info.txt"
    fields = dict(line.split(":", 1) for line in (info.read_text(encoding="utf-8").splitlines() if info.exists() else [])
                  if ":" in line)
    classes = fields.get("classes", "").replace("posidonia_oceanica", "posidonia").strip()
    label = (f"latest ({version})" if latest else version) + (f": {classes}" if classes else "")
    return label + (f" (mask mAP50 {fields['mask mAP50'].strip()})" if "mask mAP50" in fields else "")


def detect_settings():
    versions = model_versions()
    s = {
        "model": w.Dropdown(options=[(_model_label(v, latest=v == versions[-1]), v) for v in reversed(versions)],  # newest first
                            value=versions[-1],  # the latest model
                            description="Model", style=STYLE, layout=WIDE),
        "conf": _f(0.40, "Confidence", 0.05, 0.95, 0.05),
        "opacity": _f(0.30, "Colour opacity", 0.0, 1.0, 0.05),
        "saturation": _f(1.00, "Video saturation", 0.0, 2.5, 0.05),
        "every_frames": w.IntSlider(value=2, min=1, max=10, description="Every n-th frame", style=STYLE,
                                    layout=WIDE),
        "save_max": w.Dropdown(options=[("1080 px", 1080), ("1920 px", 1920), ("720 px", 720)], value=1080,
                               description="Saved video size", style=STYLE),
        "live_every": w.Dropdown(options=[("every analysed frame", 1), ("every 3rd frame", 3),
                                          ("every 10th frame", 10), ("off", 0)], value=3,
                                 description="Live view updates", style=STYLE),
    }
    display(w.VBox([w.HTML("<b>Step 2: seagrass detection</b> (YOLO11s-seg: posidonia_oceanica, rock; higher mAP = better on the "
                           "validation images) "
                           "&nbsp; <i>every n-th frame: 2 = half the frames, twice as fast</i>"), *s.values()]))
    return s


def coverage_settings():
    s = {
        "every": _f(0.5, "Seconds between frames", 0.2, 3.0, 0.1),
        "cell": w.Dropdown(options=[0.1, 0.25, 0.5, 1.0], value=0.25, description="Cell size (m)", style=STYLE),
        "max_range": _f(8.0, "Max range (m)", 3.0, 15.0, 0.5),
        "min_views": w.IntSlider(value=3, min=1, max=10, description="Min. views per cell", style=STYLE,
                                 layout=WIDE),
        "threshold": w.IntSlider(value=50, min=10, max=90, step=5, description="Class threshold (%)", style=STYLE,
                                 layout=WIDE),
        "min_area": _f(1.0, "Min. polygon (m2)", 0.0, 20.0, 0.5),
        "tilt": _f(64.0, "Camera tilt (deg)", 30.0, 90.0, 1.0),
        "vfov": _f(75.3, "Vertical FOV (deg)", 40.0, 110.0, 0.1),
        "hfov": _f(46.2, "Horizontal FOV (deg)", 20.0, 110.0, 0.1),
    }
    display(w.VBox([w.HTML("<b>Step 3: coverage map</b> &nbsp; <i>camera geometry: GoPro HERO13 4K Wide behind a "
                           "flat port, portrait (also used by step 1)</i>"), *s.values()]))
    return s


def _values(panel):
    return {k: v.value for k, v in panel.items()}


def write_config(video=None, enhance=None, detect=None, coverage=None):
    """config.ini = config_template.ini with the chosen values."""
    cfg = configparser.ConfigParser()
    target = HERE / "config.ini"
    cfg.read(target if target.exists() else HERE / "config_template.ini", encoding="utf-8")

    def put(section, key, value):
        if isinstance(value, bool):
            value = "yes" if value else "no"
        cfg[section][key] = str(value)

    if video:
        put("enhance", "video", Path(video).relative_to(HERE).as_posix())
    for key, value in (enhance or {}).items():
        if key != "run":
            put("enhance", key, value)
    if enhance and "particle_method" in enhance:
        for old in ("denoise_frames", "denoise_mode"):  # replaced by remove_particles + particle_method
            cfg.remove_option("enhance", old)
    for key, value in (detect or {}).items():
        if key != "video":
            put("detect", key, value)
    if detect and detect.get("video"):
        put("detect", "video", Path(detect["video"]).relative_to(HERE).as_posix())
    for key, value in (coverage or {}).items():
        if key in ("tilt", "vfov", "hfov"):
            put("enhance", key, value)  # the same camera for every step
        else:
            put("coverage", key, value)
    with open(target, "w", encoding="utf-8") as f:
        cfg.write(f)
    return cfg


def _read_lines(stream, lines):
    """Background thread: the script's output as (line, is_progress) into a queue; None at the end."""
    buf = b""
    while True:
        chunk = stream.read(256)
        if not chunk:
            break
        buf += chunk
        while True:
            # \r\n ends a line (Windows), a lone \r is a progress line that overwrites itself
            m = re.search(rb"\r\n|\r|\n", buf)
            if not m or (m.group() == b"\r" and m.end() == len(buf)):  # wait: may be the start of \r\n
                break
            line, sep, buf = buf[:m.start()].decode(errors="replace"), m.group(), buf[m.end():]
            if line.strip():
                lines.put((line, sep == b"\r" or bool(re.search(r"\d+/\d+ frames", line))))
    if buf.strip():
        lines.put((buf.decode(errors="replace").strip(), False))
    lines.put(None)


def run_script(script, live=None):
    """Run one of the scripts, showing its output as it comes (progress lines update in place).
    live = function called about twice a second while it runs (e.g. to show the latest frame).
    The notebook's stop button (interrupt the kernel) stops the script too."""
    import queue
    import threading
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    proc = subprocess.Popen([sys.executable, "-u", str(CODE / script)], cwd=HERE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, bufsize=0)
    lines = queue.Queue()
    threading.Thread(target=_read_lines, args=(proc.stdout, lines), daemon=True).start()
    progress = display(HTML(""), display_id=True)
    start = time.time()
    try:
        while True:
            try:
                item = lines.get(timeout=0.5)
            except queue.Empty:
                item = ()
            if item is None:
                break
            if item:
                line, is_progress = item
                if is_progress:
                    progress.update(HTML(f"<code>{line.strip()}</code> &nbsp; ({(time.time() - start) / 60:.1f} min)"))
                else:
                    print(line)
            if live:
                live()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        print(f"--- {script} stopped")
        raise
    code = proc.wait()
    if live:
        live()
    if code:
        raise RuntimeError(f"{script} stopped with an error (exit code {code}); see the output above")
    print(f"--- {script} finished in {(time.time() - start) / 60:.1f} min")


# ------------------------------------------------------------------
# step 1: enhancement
# ------------------------------------------------------------------

def run_enhance(picker, panel):
    video = INPUT / picker.value
    STATE["video"] = video
    values = _values(panel)
    run = new_run(video)  # every step 1 starts a new run folder
    if not values["run"]:
        STATE["enhanced"] = None
        print(f"Enhancement skipped: step 2 will use the original {video.name}")
        return
    folder = run / "enhancement"  # the enhanced video, its .srt, the preview and the fit chart
    cfg = write_config(video=video, enhance=dict(values, output_dir=_rel(folder)))
    _save_settings(run)
    run_script("0.enhance_video.py")
    method = _enhance_name(cfg["enhance"])
    out = folder / f"{video.stem}_enhanced_{method}.mp4"
    preview = folder / f"{video.stem}_enhance_{method}_preview.jpg"
    if preview.exists():
        print("Before (top) and after (bottom) at a few moments:")
        display(Image(filename=str(preview), width=900))
    if not values["preview_only"] and out.exists():
        STATE["enhanced"] = out
        print(f"Enhanced video: {_rel(out)}")


def _enhance_name(enh):
    """The method part of the enhanced file's name, built as 0.enhance_video.py builds it,
    e.g. natural, natural_depth, natural_lc2, natural_pmedian5, natural_depth_pbright5_lc1.5"""
    name = "natural" + ("_depth" if enh.get("range", "srt").strip().lower() == "depth_anything" else "")
    if enh.get("remove_particles", "no").strip().lower() in ("yes", "true", "on", "1"):
        method = enh.get("particle_method", "median").strip().lower()
        frames = int(float(enh.get("particle_frames", "5") or 5))
        if method in ("median", "bright") and frames > 1:
            name += f"_p{method}{frames // 2 * 2 + 1}"
        elif method == "blur":
            name += "_pblur"
    contrast = float(enh.get("local_contrast", "0") or 0)
    if contrast > 0:
        name += f"_lc{contrast:g}"
    return name


# ------------------------------------------------------------------
# step 2: detection
# ------------------------------------------------------------------

def _class_colour(i):
    from ultralytics.utils.plotting import colors
    r, g, b = colors(i, False)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _sparkline(history, colours, width=300, height=90):
    """Inline SVG: cover % of each class over the analysed frames so far."""
    if len(history) < 2:
        return ""
    lines = []
    for k, colour in enumerate(colours):
        pts = " ".join(f"{width * i / (len(history) - 1):.1f},{height - height * row[k] / 100:.1f}"
                       for i, row in enumerate(history))
        lines.append(f"<polyline points='{pts}' fill='none' stroke='{colour}' stroke-width='1.5'/>")
    return (f"<svg width='{width}' height='{height}' style='background:#f4f4f4;border:1px solid #ddd'>"
            f"{''.join(lines)}</svg><div style='font-size:11px;color:#666'>cover % per frame (0-100)</div>")


class LiveView:
    """The detection 'window' inside the notebook: latest annotated frame, progress and what it found."""

    def __init__(self, folder):
        import base64
        self.b64 = base64.b64encode
        self.jpg, self.json = folder / "live.jpg", folder / "live.json"
        for p in (self.jpg, self.json):
            p.unlink(missing_ok=True)
        self.seen = 0
        self.history = []
        self.handle = display(HTML("<i>Starting the detection: the live view appears after the first frames...</i>"),
                              display_id=True)

    def __call__(self):
        import json
        try:
            stamp = self.json.stat().st_mtime_ns
            if stamp == self.seen:
                return
            info = json.loads(self.json.read_text(encoding="utf-8"))
            picture = self.b64(self.jpg.read_bytes()).decode()
        except (OSError, ValueError):
            return  # not written yet, or being replaced right now
        self.seen = stamp
        names = list(info["classes"])
        colours = [_class_colour(i) for i in range(len(names))]
        self.history.append([info["classes"][n]["cover_pct"] for n in names])
        share = info["done"] / max(info["to_do"], 1)
        left = (info["to_do"] - info["done"]) / max(info["fps"], 1e-6)
        rows = "".join(
            f"<tr><td><span style='display:inline-block;width:12px;height:12px;background:{c};margin-right:6px'></span>"
            f"{n}</td><td style='text-align:right'>{info['classes'][n]['count']}</td>"
            f"<td style='text-align:right'>{info['classes'][n]['max_conf']:.2f}</td>"
            f"<td style='white-space:nowrap'><div style='display:inline-block;vertical-align:middle;width:110px;"
            f"background:#eee'><div style='width:{min(info['classes'][n]['cover_pct'], 100):.0f}%;"
            f"background:{c};height:10px'></div></div> {info['classes'][n]['cover_pct']:.0f} %</td></tr>"
            for n, c in zip(names, colours))
        self.handle.update(HTML(f"""
<div style="display:flex;gap:18px;align-items:flex-start;font-family:sans-serif;font-size:13px">
  <img src="data:image/jpeg;base64,{picture}" style="height:520px;border:1px solid #ccc"/>
  <div style="min-width:320px">
    <div style="font-size:15px;font-weight:bold;margin-bottom:6px">Live detection</div>
    <div style="width:300px;background:#eee;height:14px"><div style="width:{100 * share:.1f}%;background:#2a7;height:14px"></div></div>
    <div style="margin:4px 0 10px">{info['done']} / {info['to_do']} frames ({100 * share:.0f} %) &middot;
      video {info['time_s']:.1f} s &middot; {info['fps']:.1f} fps &middot; ~{left / 60:.1f} min left</div>
    <style>.lv td {{padding:3px 8px 3px 0}}</style>
    <table class="lv" style="border-collapse:collapse"><tr style="color:#666"><td>class</td><td>areas</td><td>best conf</td>
      <td>cover of the frame</td></tr>{rows}</table>
    <div style="margin-top:12px">{_sparkline(self.history, colours)}</div>
  </div>
</div>"""))


def run_detect(panel):
    video = STATE["enhanced"] or STATE["video"]
    if video is None:
        raise RuntimeError("Choose a video first (top of the notebook)")
    run = current_run(video)
    folder = run / "prediction"
    write_config(detect=dict(_values(panel), video=video, output_dir=_rel(folder)))
    _save_settings(run)
    STATE["model"] = panel["model"].value
    live = LiveView(folder) if panel["live_every"].value else None
    run_script("detect_video.py", live=live)
    name = f"{video.stem}_{STATE['model']}"
    STATE["detected"] = video
    out = folder / f"{name}_prediction.mp4"
    table = pd.read_csv(folder / f"{name}_detections.csv")
    print("Some frames of the annotated video:")
    show_detection_frames(out)
    fig, ax = plt.subplots(figsize=(10, 3))
    for col in [c for c in table.columns if c.endswith("_cover_pct")]:
        ax.plot(table.time_s, table[col], label=col.replace("_cover_pct", ""))
    ax.set_xlabel("video time (s)")
    ax.set_ylabel("% of the frame")
    ax.set_title("Detected cover per frame")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("Average % of the frame: " + ", ".join(f"{c.replace('_cover_pct', '')} {table[c].mean():.1f} %"
                                                 for c in table.columns if c.endswith("_cover_pct")))
    print(f"Annotated video: {out.relative_to(HERE)} (play it below, or download it)")
    display(Video(out.relative_to(HERE).as_posix(), width=360, html_attributes="controls muted"))


def browse_detections(video=None, model=None, run=None):
    """Step through the annotated video frame by frame (slider or play button), with the numbers
    of each frame and its place on the cover chart. Without arguments: the last detection of this
    session; earlier results e.g. browse_detections("GX010287_sample_enhanced_natural", "v13")
    (the newest run that has them, or run="2026-11-08_101532_GX010287_sample")."""
    video = Path(video) if video else STATE["detected"]
    if video is None:
        raise RuntimeError("Run the detection (step 2) first")
    model = model or STATE["model"] or model_versions()[-1]
    name = f"{video.stem}_{model}"
    path = _find("prediction", f"{name}_prediction.mp4", run)
    table = pd.read_csv(path.parent / f"{name}_detections.csv")
    names = [c[:-len("_count")] for c in table.columns if c.endswith("_count")]
    colours = [_class_colour(i) for i in range(len(names))]
    cap = cv2.VideoCapture(str(path))
    n = min(len(table), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15

    picture = w.Image(format="jpeg", layout=w.Layout(height="560px"))
    numbers = w.HTML()
    chart = w.Output()
    slider = w.IntSlider(value=0, min=0, max=n - 1, description="Frame", continuous_update=True,
                         layout=w.Layout(width="520px"))
    play = w.Play(value=0, min=0, max=n - 1, interval=int(1000 / fps), description="Play")
    w.jslink((play, "value"), (slider, "value"))
    state = {"next": 0}

    with chart:
        fig, ax = plt.subplots(figsize=(5.2, 2.2))
        for name_, colour in zip(names, colours):
            ax.plot(table.time_s[:n], table[f"{name_}_cover_pct"][:n], color=colour, label=name_, lw=1)
        marker = ax.axvline(table.time_s[0], color="k", lw=1)
        ax.set_xlabel("video time (s)", fontsize=8)
        ax.set_ylabel("% of the frame", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    chart_fig = fig

    def show(change=None):
        k = slider.value
        if k != state["next"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)  # jump; plain reads when playing forward
        ok, frame = cap.read()
        state["next"] = k + 1
        if ok:
            picture.value = cv2.imencode(".jpg", fit_side(frame, 720), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        r = table.iloc[k]
        rows = "".join(
            f"<tr><td><span style='display:inline-block;width:12px;height:12px;background:{c};margin-right:6px'></span>"
            f"{nm}</td><td style='text-align:right'>{int(r[f'{nm}_count'])}</td>"
            f"<td style='text-align:right'>{r[f'{nm}_max_conf']:.2f}</td>"
            f"<td style='text-align:right'>{r[f'{nm}_cover_pct']:.1f} %</td></tr>" for nm, c in zip(names, colours))
        numbers.value = (f"<div style='font-family:sans-serif;font-size:13px'><b>video {r.time_s:.2f} s</b> "
                         f"(frame {int(r.frame)} of the video)<br><table style='margin-top:6px'>"
                         f"<tr style='color:#666'><td>class</td><td>areas</td><td>best conf</td><td>cover</td></tr>"
                         f"{rows}</table></div>")
        marker.set_xdata([r.time_s, r.time_s])
        with chart:
            chart.clear_output(wait=True)
            display(chart_fig)

    slider.observe(show, names="value")
    show()
    display(w.HBox([picture, w.VBox([w.HBox([play, slider]), numbers, chart])]))


def fit_side(img, max_side):
    h, wd = img.shape[:2]
    s = max_side / max(h, wd)
    return cv2.resize(img, (int(wd * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def show_detection_frames(path, n=4):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 6))
    for ax, i in zip(np.atleast_1d(axes), np.linspace(0.1, 0.9, n) * total):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            ax.imshow(f[..., ::-1])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# step 3: coverage map
# ------------------------------------------------------------------

def run_coverage(panel):
    video = STATE["detected"] or STATE["enhanced"] or STATE["video"]
    if video is None:
        raise RuntimeError("Choose a video first (top of the notebook)")
    if not video.with_suffix(".srt").exists():
        raise RuntimeError(f"The coverage map needs {video.with_suffix('.srt').name} (GPS + altitude) next to the video")
    run = current_run(video)
    cfg = write_config(coverage=dict(_values(panel), video=_rel(video), model="", conf="",
                                     output_dir=_rel(run / "coverage")))
    _save_settings(run)
    run_script("6.make_coverage_map.py")
    model = STATE["model"] or cfg["detect"].get("model", "latest").strip()  # the model of step 2
    model = model_versions()[-1] if model.lower() == "latest" else model
    STATE["coverage_name"] = f"{video.stem}_{model}"
    show_map()


def show_map(name=None, run=None):
    """Coverage rasters, polygons and the camera track: a static figure and an interactive map.
    Without arguments: the last map of this session; earlier ones e.g.
    show_map("GX010287_sample_enhanced_natural_v13") (<video>_<model>; the newest run that has it)."""
    import geopandas as gpd
    import rasterio
    name = name or STATE["coverage_name"]
    gpkg = _find("coverage", f"{name}_coverage.gpkg", run)
    folder = gpkg.parent
    rasters = sorted(folder.glob(f"{name}_*_coverage.tif"))
    classes = [p.name[len(name) + 1:-len("_coverage.tif")] for p in rasters]
    fig, axes = plt.subplots(1, len(rasters) + 1, figsize=(5 * (len(rasters) + 1), 6))
    with rasterio.open(folder / f"{name}_mosaic.tif") as src:
        b = src.bounds
        axes[0].imshow(np.moveaxis(src.read(), 0, -1), extent=(b.left, b.right, b.bottom, b.top))
        axes[0].set_title("frames on the map (check)")
    track = gpd.read_file(gpkg, layer="camera_track")
    for ax, path, cls in zip(axes[1:], rasters, classes):
        with rasterio.open(path) as src:
            data = np.ma.masked_less(src.read(1), 0)
            b = src.bounds
        cmap = {"posidonia_oceanica": "YlGn", "rock": "copper_r", "debris": "Oranges"}.get(cls, "Blues")
        im = ax.imshow(data, extent=(b.left, b.right, b.bottom, b.top), cmap=cmap, vmin=0, vmax=100)
        gpd.read_file(gpkg, layer=cls).boundary.plot(ax=ax, color="k", linewidth=0.6)
        fig.colorbar(im, ax=ax, shrink=0.6, label="% of the views")
        ax.set_title(f"{cls} cover")
    for ax in axes:
        track.plot(ax=ax, color="red", linewidth=1)
        ax.set_xlabel("east (m)")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("north (m)")
    plt.tight_layout()
    plt.show()

    surveyed = gpd.read_file(gpkg, layer="surveyed_area").area.sum()
    rows = []
    for cls in classes:
        p = gpd.read_file(gpkg, layer=cls)
        rows.append({"class": cls, "patches": len(p), "area_m2": round(p.area.sum(), 1),
                     "share_of_surveyed_%": round(100 * p.area.sum() / max(surveyed, 1), 1)})
    print(f"Surveyed area: {surveyed:,.0f} m2")
    display(pd.DataFrame(rows))
    interactive_map(gpkg, classes)


# ------------------------------------------------------------------
# QGIS styles on the interactive map: the styles stored in the GeoPackage (layer_styles, the same
# ones QGIS opens the layers with) are translated to Leaflet. Each QGIS symbol layer becomes one
# drawing of the features, stacked in the same order: plain fills and outlines as Leaflet styles,
# hatching, textures and random markers (the seagrass blades) as SVG patterns, centroid markers
# as small SVG icons. Sizes in QGIS millimetres are drawn at 96 dpi, so like in QGIS the patterns
# keep their size on screen at every zoom.
# ------------------------------------------------------------------

MM = 96 / 25.4  # QGIS millimetres -> screen pixels
MAP_FALLBACK = {  # layers without a stored style
    "surveyed_area": {"color": "#555", "weight": 1, "fillOpacity": 0.05},
    "camera_track": {"color": "red", "weight": 2},
    "posidonia_oceanica": {"color": "#1a9850", "weight": 1, "fillColor": "#1a9850", "fillOpacity": 0.5},
    "rock": {"color": "#8c510a", "weight": 1, "fillColor": "#8c510a", "fillOpacity": 0.5},
    "debris": {"color": "#f06e14", "weight": 1, "fillColor": "#f06e14", "fillOpacity": 0.5},
}


def _stored_styles(gpkg):
    """layer name -> QGIS style (QML): the GeoPackage's default styles, else styles/<layer>.qml"""
    import sqlite3
    found = {}
    try:
        with sqlite3.connect(gpkg) as db:
            for name, qml in db.execute("SELECT f_table_name, styleQML FROM layer_styles WHERE useAsDefault "
                                        "ORDER BY update_time"):
                found[name] = qml
    except sqlite3.Error:
        pass  # results from before the styles were stored
    for path in (HERE / "styles").glob("*.qml"):
        found.setdefault(path.stem, path.read_text(encoding="utf-8"))
    return found


def _qml_layers(qml):
    """[(class, options, [(sub class, sub options, data-defined expressions)])] of a single-symbol style"""
    import xml.etree.ElementTree as ET

    def options(el):
        return {o.get("name"): o.get("value") for o in el.findall("./Option/Option") if o.get("value") is not None}

    def defined(el):
        out = {}
        for prop in el.findall("./data_defined_properties/Option/Option[@name='properties']/Option"):
            o = {x.get("name"): x.get("value") for x in prop.findall("./Option")}
            if o.get("active") == "true" and o.get("expression"):
                out[prop.get("name")] = o["expression"]
        return out

    symbol = ET.fromstring(qml).find("./renderer-v2/symbols/symbol")
    if symbol is None:
        return []
    return [(layer.get("class"), options(layer),
             [(s.get("class"), options(s), defined(s)) for sub in layer.findall("./symbol")
              for s in sub.findall("./layer") if s.get("enabled", "1") == "1"])
            for layer in symbol.findall("./layer") if layer.get("enabled", "1") == "1"]


def _rgba(value, default="0,0,0,255"):
    r, g, b, a = (int(x) for x in (value or default).split(",")[:4])
    return f"#{r:02x}{g:02x}{b:02x}", round(a / 255, 3)


def _px(o, key, default):
    """a QGIS size in mm (or pixels) -> screen pixels"""
    value = float(o.get(key) or default)
    return value if o.get(f"{key}_unit", "MM") == "Pixel" else value * MM


def _stroke(o, colour_key, width_key, style_key):
    """Leaflet stroke options of a QGIS outline / line"""
    style = o.get(style_key, "solid")
    if style == "no":
        return {"stroke": False}
    colour, alpha = _rgba(o.get(colour_key))
    w = max(_px(o, width_key, 0.26), 1.0)
    dash = None
    if o.get("use_custom_dash") == "1" and o.get("customdash"):
        k = 1 if o.get("customdash_unit") == "Pixel" else MM
        dash = " ".join(f"{float(v) * k:.1f}" for v in o["customdash"].split(";"))
    elif style != "solid":
        units = {"dash": (4, 2), "dot": (1, 2), "dash dot": (4, 2, 1, 2), "dash dot dot": (4, 2, 1, 2, 1, 2)}.get(style)
        dash = " ".join(f"{u * w:.1f}" for u in units) if units else None
    out = {"stroke": True, "color": colour, "opacity": alpha, "weight": round(w, 2),
           "lineCap": {"square": "square", "round": "round", "flat": "butt"}.get(o.get("capstyle"), "round")}
    return dict(out, dashArray=dash) if dash else out


def _rand(expression, default):
    """'rand(-25,25)' or 'rand(192,416)/100' -> (low, high); anything else -> (default, default)"""
    m = re.fullmatch(r"\s*rand\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s*(?:/\s*([\d.]+))?\s*", expression or "")
    if not m:
        return default, default
    k = float(m.group(3) or 1)
    return float(m.group(1)) / k, float(m.group(2)) / k


def _random_marker_tile(key, o, sub, tile=64.0):
    """RandomMarkerFill -> a seamless SVG pattern tile with the markers (seagrass blades: 'line')"""
    so, expr = sub[1], sub[2]
    area = (tile / MM) ** 2  # mm2
    density = float(o.get("point_count") or 10) / max(float(o.get("density_area") or 250), 1e-6)
    n = int(round(density * area)) if o.get("count_method") == "1" else int(o.get("point_count") or 10)
    rng = np.random.default_rng(int(float(o.get("seed") or 0)))
    angles, sizes = _rand(expr.get("angle"), float(so.get("angle") or 0)), _rand(expr.get("size"), float(so.get("size") or 2))
    line = so.get("name") == "line"
    colour, alpha = _rgba(so.get("outline_color" if line else "color"))
    width = max(_px(so, "outline_width", 0.2), 0.5)
    items = []
    for _ in range(n):
        x, y = rng.uniform(0, tile, 2)
        angle, size = np.radians(rng.uniform(*angles)), rng.uniform(*sizes) * MM
        dx, dy = np.sin(angle) * size / 2, -np.cos(angle) * size / 2  # QGIS 'line' = vertical, turned clockwise
        for ox in (-tile, 0, tile):  # also drawn one tile over where it crosses an edge: no seams
            for oy in (-tile, 0, tile):
                if -size < x + ox < tile + size and -size < y + oy < tile + size:
                    cx, cy = x + ox, y + oy
                    items.append(f'<line x1="{cx - dx:.1f}" y1="{cy - dy:.1f}" x2="{cx + dx:.1f}" y2="{cy + dy:.1f}"/>' if line
                                 else f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{size / 2:.1f}"/>')
    paint = (f'stroke="{colour}" stroke-opacity="{alpha}" stroke-width="{width:.2f}" stroke-linecap="square"' if line
             else f'fill="{colour}" fill-opacity="{alpha}"')
    return (f'<pattern id="{key}" patternUnits="userSpaceOnUse" width="{tile:g}" height="{tile:g}">'
            f'<g {paint}>{"".join(items)}</g></pattern>')


def _web_layers(key, qml, defs):
    """A QGIS style -> [("style", Leaflet style) | ("markers", marker spec)], bottom first; SVG patterns
    are added to defs"""
    import base64
    out = []
    for i, (cls, o, subs) in enumerate(_qml_layers(qml)):
        pid = f"qgis_{key}_{i}"
        if cls == "SimpleFill":
            style = {"stroke": False, "fillOpacity": 0}
            if o.get("style", "solid") != "no":
                colour, alpha = _rgba(o.get("color"))
                style.update(fillColor=colour, fillOpacity=alpha)
            out.append(("style", dict(style, **_stroke(o, "outline_color", "outline_width", "outline_style"))))
        elif cls == "SimpleLine":  # a line layer, or an outline drawn over a polygon's fill
            out.append(("style", dict({"fill": False}, **_stroke(o, "line_color", "line_width", "line_style"))))
        elif cls == "LinePatternFill" and subs:
            d = max(_px(o, "distance", 2), 2.0)
            colour, alpha = _rgba(subs[0][1].get("line_color"))
            w = max(_px(subs[0][1], "line_width", 0.26), 0.5)
            defs.append(f'<pattern id="{pid}" patternUnits="userSpaceOnUse" width="{d:.2f}" height="{d:.2f}" '
                        f'patternTransform="rotate({-float(o.get("angle") or o.get("lineangle") or 0):g})">'
                        f'<line x1="-1" y1="{d / 2:.2f}" x2="{d + 1:.2f}" y2="{d / 2:.2f}" stroke="{colour}" '
                        f'stroke-opacity="{alpha}" stroke-width="{w:.2f}"/></pattern>')
            out.append(("style", {"stroke": False, "fillColor": f"url(#{pid})", "fillOpacity": 1}))
        elif cls == "RasterFill" and (o.get("imageFile") or "").startswith("base64:"):
            data = o["imageFile"][len("base64:"):]
            img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            w = _px(o, "width", 0) or img.shape[1]
            h = w * img.shape[0] / img.shape[1]
            mime = "png" if data.startswith("iVBOR") else "jpeg"
            defs.append(f'<pattern id="{pid}" patternUnits="userSpaceOnUse" width="{w:.2f}" height="{h:.2f}">'
                        f'<image href="data:image/{mime};base64,{data}" width="{w:.2f}" height="{h:.2f}" '
                        f'preserveAspectRatio="none"/></pattern>')
            out.append(("style", {"stroke": False, "fillColor": f"url(#{pid})", "fillOpacity": float(o.get("alpha") or 1)}))
        elif cls == "RandomMarkerFill" and subs:
            defs.append(_random_marker_tile(pid, o, subs[0]))
            out.append(("style", {"stroke": False, "fillColor": f"url(#{pid})", "fillOpacity": 1}))
        elif cls == "CentroidFill" and subs:
            so = subs[0][1]
            out.append(("markers", {"shape": so.get("name", "circle"), "fill": _rgba(so.get("color")),
                                    "line": _rgba(so.get("outline_color")), "width": _px(so, "outline_width", 0.2),
                                    "size": _px(so, "size", 2)}))
    return out


def _marker_icon(spec):
    import folium
    s, (fc, fa), (lc, la), w = spec["size"], spec["fill"], spec["line"], spec["width"]
    paint = f'fill="{fc}" fill-opacity="{fa}" stroke="{lc}" stroke-opacity="{la}" stroke-width="{w:.2f}"'
    shape = {"triangle": f'<polygon points="{s / 2:.1f},0 {s:.1f},{s * 0.87:.1f} 0,{s * 0.87:.1f}" {paint}/>',
             "square": f'<rect width="{s:.1f}" height="{s:.1f}" {paint}/>'}.get(
        spec["shape"], f'<circle cx="{s / 2:.1f}" cy="{s / 2:.1f}" r="{s / 2:.1f}" {paint}/>')
    html = f'<svg width="{s:.1f}" height="{s:.1f}" style="overflow:visible">{shape}</svg>'
    return folium.DivIcon(html=html, icon_size=(s, s), icon_anchor=(s / 2, s / 2), class_name="qgis-marker")


def interactive_map(gpkg, classes):
    try:
        import folium
        import geopandas as gpd
    except ImportError:
        return
    track = gpd.read_file(gpkg, layer="camera_track").to_crs(4326)
    centre = track.geometry.iloc[0].centroid
    m = folium.Map(location=[centre.y, centre.x], zoom_start=19, max_zoom=23, tiles=None)
    folium.TileLayer("OpenStreetMap", max_zoom=23, max_native_zoom=19).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Satellite", max_zoom=23,
                     max_native_zoom=18).add_to(m)  # Esri has photos to level 18 here (19+ = "Map data not yet available")
    styles, defs = _stored_styles(gpkg), []
    for name in ["surveyed_area", *classes, "camera_track"]:  # bottom to top
        layer = gpd.read_file(gpkg, layer=name).to_crs(4326)
        if not len(layer):
            continue
        parts = _web_layers(re.sub(r"\W", "_", name), styles[name], defs) if name in styles else []
        parts = parts or [("style", MAP_FALLBACK.get(name, {"color": "#3366cc", "weight": 1, "fillOpacity": 0.5}))]
        group = folium.FeatureGroup(name=name.replace("_", " "))
        last = max(i for i, (kind, _) in enumerate(parts) if kind == "style") if any(k == "style" for k, _ in parts) else -1
        for i, (kind, spec) in enumerate(parts):
            if kind == "style":
                tip = folium.GeoJsonTooltip(["class", "area_m2"]) if i == last and name in classes else None
                folium.GeoJson(layer, style_function=lambda f, s=spec: s, tooltip=tip).add_to(group)
            else:  # a marker at each patch's centre (QGIS CentroidFill)
                for geom in layer.geometry:
                    for part in getattr(geom, "geoms", [geom]):
                        c = part.centroid
                        folium.Marker([c.y, c.x], icon=_marker_icon(spec)).add_to(group)
        group.add_to(m)
    if defs:  # the patterns, in an SVG of their own in the page (Leaflet's paths refer to them by id)
        m.get_root().html.add_child(folium.Element(
            '<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>' + "".join(defs) + "</defs></svg>"
            "<style>.qgis-marker{background:none;border:none}</style>"))
    folium.LayerControl().add_to(m)
    b = track.total_bounds
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    display(m)


# ------------------------------------------------------------------
# download
# ------------------------------------------------------------------

# what can be chosen for the zip: kind -> which files belong to it
ZIP_KINDS = {
    "enhanced videos": lambda p: ("_enhanced_" in p.name and p.suffix.lower() in (".mp4", ".srt")
                                  and p.parent.name != "prediction"),  # the detection video's name has _enhanced_ too
    "enhancement previews and charts": lambda p: p.parent.name == "enhancement" and "_enhanced_" not in p.name,
    "detection videos": lambda p: p.name.endswith("_prediction.mp4"),
    "detection tables (CSV)": lambda p: p.name.endswith("_detections.csv"),
    "coverage maps (GeoTIFF, GeoPackage, shapefiles)": lambda p: p.parent.name == "coverage",
    "settings used": lambda p: p.name == "settings.ini",
}


def _old_results():
    """Files from before the runs had their own folders: results/<step>/ and input/*_enhanced_*."""
    files = [p for sub in ("enhancement", "prediction", "coverage") if (RESULTS / sub).is_dir()
             for p in (RESULTS / sub).iterdir() if p.is_file()]
    return files + sorted(INPUT.glob("*_enhanced_*"))


def _run_label(run):
    """'2026-11-08 10:15 GX010175_sample: enhanced, detected (v13), mapped'"""
    made = []
    if any(run.glob("enhancement/*_enhanced_*.mp4")):
        made.append("enhanced")
    models = sorted({p.stem.rsplit("_", 2)[-2] for p in run.glob("prediction/*_prediction.mp4")})
    if models:
        made.append(f"detected ({', '.join(models)})")
    if any(run.glob("coverage/*_coverage.gpkg")):
        made.append("mapped")
    n = run.name
    return f"{n[:10]} {n[11:13]}:{n[13:15]} {n[18:]}: " + (", ".join(made) or "started")


def zip_files(picked_runs, picked_kinds):
    """The files for the zip: picked_runs = run folder names, "_old" or "all"; picked_kinds = keys
    of ZIP_KINDS or "all"."""
    picked = set(picked_runs)
    files = []
    for r in runs():
        if "all" in picked or r.name in picked:
            files += [p for p in r.rglob("*") if p.is_file()]
    if "all" in picked or "_old" in picked:
        files += _old_results()
    files = [p for p in files if not p.name.startswith("live.") and p.suffix != ".tmp"]
    wanted = list(ZIP_KINDS) if "all" in picked_kinds else list(picked_kinds)
    return [p for p in files if any(ZIP_KINDS[k](p) for k in wanted)]


def zip_results(include_videos=True):
    """Panel: choose the runs and what to put in results.zip, then download it (Binder sessions are
    deleted when closed). include_videos=False starts with the videos left out."""
    from IPython.display import FileLink
    found, old = runs(), _old_results()
    options = [("all", "all")] + [(_run_label(r), r.name) for r in found]
    if old:
        options.append(("results from before the run folders", "_old"))
    run_pick = w.SelectMultiple(options=options, value=(found[0].name,) if found else ("all",),
                                rows=min(8, len(options)), description="Runs", style=STYLE,
                                layout=w.Layout(width="760px"))
    kinds = list(ZIP_KINDS)
    kind_pick = w.SelectMultiple(options=["all"] + kinds,
                                 value=("all",) if include_videos else tuple(k for k in kinds if "video" not in k),
                                 rows=len(kinds) + 1, description="What", style=STYLE, layout=w.Layout(width="760px"))
    button = w.Button(description="Make the zip", icon="download", button_style="primary")
    out = w.Output()

    def make(_):
        out.clear_output()
        with out:
            files = zip_files(run_pick.value, kind_pick.value)
            if not files:
                print("Nothing to zip for this choice.")
                return
            target = HERE / "results.zip"
            print(f"Packing {len(files)} files ...")
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
                for p in files:
                    z.write(p, p.relative_to(HERE))
            print(f"{target.name}: {len(files)} files, {target.stat().st_size / 1e6:.1f} MB")
            if "google.colab" in sys.modules:
                # Colab doesn't serve the notebook's folder, so a link leads nowhere: send the file
                # with Colab's own download (the browser may ask to allow downloads from the page)
                from google.colab import files
                print(f"Downloading... (or right-click {target.relative_to(HERE.parent).as_posix()} "
                      "in the Files panel on the left > Download)")
                files.download(str(target))
            else:
                display(FileLink(target.name, result_html_prefix="Download: "))

    button.on_click(make)
    display(w.VBox([w.HTML("<b>Download the results</b> &nbsp; <i>choose the runs and what to include "
                           "(Ctrl + click or Shift + click to choose several), then press the button</i>"),
                    run_pick, kind_pick, button, out]))
