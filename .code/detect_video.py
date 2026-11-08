"""Step 2: detect the seagrass in a video and save the annotated video (no window, for Binder /
servers; the desktop version with a live window is 4.live_detect.py in the full project).

Reads [detect] in config.ini. Saves to output_dir (default results/prediction/; the notebook
gives each run its own folder, results/<run>/prediction/):
  <video>_<model>_prediction.mp4  the video with the detected areas coloured, boxes and confidence
  <video>_<model>_detections.csv  one row per analysed frame: per class the number of areas, the
                                  highest confidence and the % of the frame covered"""
import configparser
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import RTDETR, YOLO
from ultralytics.utils.plotting import colors

# The project folder: this script's folder, or its parent when the script sits in the
# hidden .code/ folder (Binder version)
ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == ".code" else ROOT
config = configparser.ConfigParser()
config.read(ROOT / "config.ini", encoding="utf-8")
cfg = config["detect"]

VIDEO = Path(cfg["video"])
CONF = cfg.getfloat("conf")
SAVE_MAX = cfg.getint("save_max", 1080)  # longest side of the saved video, 0 = full resolution
OPACITY = cfg.getfloat("opacity", 0.3)
SATURATION = cfg.getfloat("saturation", 1.0)  # colour boost of the saved video / live view only, 1 = unchanged
EVERY = max(1, cfg.getint("every_frames", 1))  # run the model on every n-th frame
LIVE_EVERY = cfg.getint("live_every", 3)  # every this many analysed frames the live view is updated, 0 = off
LIVE_SIZE = 720  # longest side of the live view picture


def find_model(choice):
    """'latest', a version like 'v8', or a path to a .pt file."""
    models = Path("models")
    if choice.lower() == "latest":
        versions = [d for d in models.glob("v*") if d.name[1:].isdigit() and (d / "model.pt").exists()]
        if not versions:
            raise RuntimeError("No models in models/")
        return max(versions, key=lambda d: int(d.name[1:])) / "model.pt"
    if (models / choice / "model.pt").exists():
        return models / choice / "model.pt"
    if Path(choice).exists():
        return Path(choice)
    raise RuntimeError(f"Model not found: {choice}")


def load_detector(path):
    """YOLO or RT-DETR, depending on what the .pt file contains."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ckpt.get("ema") or ckpt.get("model")
    return RTDETR(path) if "RTDETR" in type(net).__name__ else YOLO(path)


MODEL = find_model(cfg.get("model", "latest"))
VERSION = MODEL.parent.name if MODEL.name == "model.pt" else MODEL.stem
NAME = f"{VIDEO.stem}_{VERSION}"
OUT_DIR = Path(cfg.get("output_dir", "").strip() or Path("results") / "prediction")
OUTPUT = OUT_DIR / f"{NAME}_prediction.mp4"
DETECTIONS = OUT_DIR / f"{NAME}_detections.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
# live view for the notebook: the latest annotated frame and its numbers, replaced as the
# detection goes (the notebook shows them while this script runs)
LIVE_JPG = OUTPUT.parent / "live.jpg"
LIVE_JSON = OUTPUT.parent / "live.json"

model = load_detector(MODEL)
CLASS_NAMES = list(model.names.values())
print(f"Video: {VIDEO}")
print(f"Model: {MODEL} ({type(model).__name__}), classes {', '.join(CLASS_NAMES)}")

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
to_do = (total + EVERY - 1) // EVERY
print(f"{total} frames at {fps:.2f} fps; the model runs on every {EVERY}. frame ({to_do} frames)")


def fit(img, max_side):
    h, w = img.shape[:2]
    scale = max_side / max(h, w) if max_side else 1
    if scale < 1:
        img = cv2.resize(img, (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
    return img


def saturate(img, factor):
    """More (factor > 1) or less colour, same brightness: each pixel is moved away from its own grey."""
    if factor == 1:
        return img
    f = img.astype(np.float32)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    return np.clip(grey + (f - grey) * factor, 0, 255).astype(np.uint8)


def draw(frame, result):
    """Semi-transparent class colours with an outline, then Ultralytics' boxes and labels on top.
    Also returns the % of the frame each class covers."""
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)
    share = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = np.zeros((h, w), np.uint8)
        if result.masks is not None:
            for poly, cls in zip(result.masks.xy, result.boxes.cls.tolist()):
                if int(cls) == i and len(poly) >= 3:
                    cv2.fillPoly(mask, [np.asarray(poly).astype(np.int32)], 1)
        share[name] = 100 * mask.mean()
        if mask.any():
            color = np.array(colors(i, True), np.float32)
            m = mask.astype(bool)
            out[m] = out[m] * (1 - OPACITY) + color * OPACITY
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, colors(i, True), max(1, h // 600), cv2.LINE_AA)
    return result.plot(img=out.astype(np.uint8), masks=False), share


def video_writer(path, fps, w, h):
    """H.264 through ffmpeg (plays in the browser); OpenCV's mp4v as a fallback."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.Popen(
            [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
             "-i", "-", "-vf", "scale=out_color_matrix=bt601:out_range=limited,format=yuv420p",
             "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-movflags", "+faststart",
             "-colorspace", "smpte170m", "-color_range", "tv", str(path)],
            stdin=subprocess.PIPE)
        return lambda f: proc.stdin.write(np.ascontiguousarray(f).data), lambda: (proc.stdin.close(), proc.wait())
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    return writer.write, writer.release


def swap(src, dst, tries=20):
    """os.replace that survives Windows refusing while the notebook is reading dst; the live view is optional."""
    for _ in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.01)
    src.unlink(missing_ok=True)  # skip this live update rather than stop the detection


csv_file = open(DETECTIONS, "w", newline="", encoding="utf-8")
table = csv.writer(csv_file)
table.writerow(["frame", "time_s"] + [f"{name}_{col}" for name in CLASS_NAMES for col in ("count", "max_conf", "cover_pct")])

write = close = None
start = time.time()
index = done = 0
while True:
    if index % EVERY:
        if not cap.grab():
            break
        index += 1
        continue
    ok, frame = cap.read()
    if not ok:
        break
    frame = fit(frame, SAVE_MAX)
    result = model.predict(frame, conf=CONF, imgsz=640, verbose=False)[0]
    annotated, share = draw(saturate(frame, SATURATION), result)  # the model saw the unchanged frame
    row = [index, f"{index / fps:.2f}"]
    for i, name in enumerate(CLASS_NAMES):
        confs = [c for c, k in zip(result.boxes.conf.tolist(), result.boxes.cls.tolist()) if int(k) == i]
        row += [len(confs), f"{max(confs, default=0):.2f}", f"{share[name]:.2f}"]
    table.writerow(row)
    if write is None:
        h, w = annotated.shape[:2]
        write, close = video_writer(OUTPUT, fps / EVERY, w, h)
    write(annotated)
    if LIVE_EVERY and (done % LIVE_EVERY == 0 or done + 1 == to_do):
        ok, jpg = cv2.imencode(".jpg", fit(annotated, LIVE_SIZE), [cv2.IMWRITE_JPEG_QUALITY, 80])
        LIVE_JPG.with_suffix(".tmp").write_bytes(jpg.tobytes())
        swap(LIVE_JPG.with_suffix(".tmp"), LIVE_JPG)  # whole files only, never half-written
        elapsed = time.time() - start
        info = {"frame": index, "time_s": index / fps, "done": done + 1, "to_do": to_do,
                "fps": (done + 1) / max(elapsed, 1e-6), "elapsed_s": elapsed,
                "classes": {name: {"count": row[2 + 3 * i], "max_conf": float(row[3 + 3 * i]),
                                   "cover_pct": float(row[4 + 3 * i])} for i, name in enumerate(CLASS_NAMES)}}
        LIVE_JSON.with_suffix(".tmp2").write_text(json.dumps(info), encoding="utf-8")
        swap(LIVE_JSON.with_suffix(".tmp2"), LIVE_JSON)
    index += 1
    done += 1
    if done % 25 == 0 or done == to_do:
        speed = done / (time.time() - start)
        print(f"{done}/{to_do} frames, {speed:.1f} fps, ~{(to_do - done) / speed / 60:.1f} min left", flush=True)

cap.release()
csv_file.close()
if close:
    close()
print(f"Saved annotated video: {OUTPUT}")
print(f"Saved per-frame detections: {DETECTIONS}")
print(f"Processed {done} frames in {(time.time() - start) / 60:.1f} min")
