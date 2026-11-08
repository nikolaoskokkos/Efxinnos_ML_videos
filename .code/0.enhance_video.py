import configparser
import json
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
import cv2
import matplotlib
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from lib.srt_log import describe, read_srt, write_srt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The project folder: this script's folder, or its parent when the script sits in the
# hidden .code/ folder (Binder version)
ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == ".code" else ROOT
config = configparser.ConfigParser()
config.read(ROOT / "config.ini", encoding="utf-8")
cfg = config["enhance"]

VIDEO = Path(cfg["video"])
SRT = VIDEO.with_suffix(".srt")
# the notebook's version has the natural method only; the full project's 0.enhance_video.py also
# has motion, physics, seathru and the trained networks (WaterNet, FUnIE-GAN, ...)
METHOD = cfg.get("method", "natural").strip().lower()
if METHOD != "natural":
    raise ValueError(f"method = {METHOD}: this script has the natural method only (the others are in the full project)")
TILT = cfg.getfloat("tilt")  # degrees below horizontal (centre of the image)
VFOV = cfg.getfloat("vfov")  # degrees, top to bottom of the image
ALTITUDE = cfg.getfloat("altitude")  # used when the .srt has no alt.
DEPTH_CORRECTION = cfg.getboolean("depth_correction")
CALIB_FRAMES = cfg.getint("calib_frames")
STRENGTH = cfg.getfloat("strength")
OUTPUT_MAX = cfg.getint("output_max")
PREVIEW_ONLY = cfg.getboolean("preview_only")
SAND_BRIGHTNESS = cfg.getfloat("sand_brightness", 80)  # natural: Lab L of sand / pale rock, 0 = full stretch
# Particle removal: the floating particles (the correction turns them into orange streaks).
# remove_particles = yes / no turns it on; particle_method says how:
#   median = the aligned neighbouring frames (they show the same seabed a moment apart, while a
#            particle has drifted) and the middle value of every pixel: most removed, slightly
#            softer seabed, fish fade (GX010175: ~9 in 10 moving particles gone)
#   bright = the same frames, but only what is brighter than in them is replaced: sharper, more left
#   blur   = streaks found in the frame itself, filled with the colour around them: fast, smudges
REMOVE_PARTICLES = cfg.getboolean("remove_particles", False)
PARTICLE_METHOD = cfg.get("particle_method", "").strip().lower()
PARTICLE_FRAMES = cfg.getint("particle_frames", 5)  # median / bright: frames combined, 5 = 2 before + 2 after
if not PARTICLE_METHOD:  # older settings files
    if cfg.getint("denoise_frames", 0) > 1:  # denoise_frames / denoise_mode were the neighbour-frame ways
        REMOVE_PARTICLES, PARTICLE_FRAMES = True, cfg.getint("denoise_frames")
        PARTICLE_METHOD = "bright" if cfg.get("denoise_mode", "all").strip().lower() == "particles" else "median"
    else:
        PARTICLE_METHOD = "blur"  # remove_particles = yes alone meant the blur fill
if PARTICLE_METHOD not in ("median", "bright", "blur"):
    raise ValueError("particle_method must be median, bright or blur")
if not REMOVE_PARTICLES:
    PARTICLE_METHOD = "off"
NEIGHBOURS = PARTICLE_METHOD in ("median", "bright") and PARTICLE_FRAMES > 1
DENOISE_FRAMES = PARTICLE_FRAMES if NEIGHBOURS else 0  # frames combined per output frame
# bright: brightness (0-255, in the original) above the aligned neighbouring frames that counts as
# a particle. Was 10: faint streaks 4-10 above were left, and the correction (red ~2x) turned them
# into thin orange lines (GX010175); 5 catches them, 3 hardly more but softens the seabed.
PARTICLE_STEP = cfg.getfloat("particle_step", 5.0)
# ... and how far around each found particle is replaced too (px, odd): 9 also covers the soft edges
PARTICLE_GROW = max(1, cfg.getint("particle_grow", 9)) // 2 * 2 + 1
LOCAL_CONTRAST = cfg.getfloat("local_contrast", 0.0)  # CLAHE strength on brightness, 0 = off
RANGE_BALANCE = cfg.getfloat("range_balance", 0.0)  # distance colour balance far -> near, 0 = off, 1 = full
SHADOW_BALANCE = cfg.getboolean("shadow_balance", False)  # natural: keep shadows / seagrass from going pink
HAZE_FIT = cfg.get("haze_fit", "darkest").strip().lower()  # natural: darkest (per channel) or colour (one water colour)
FAR_COLOUR_LIMIT = cfg.getfloat("far_colour_limit", 0.0)  # natural: max colour change with range per channel (e.g. 1.3), 0 = off
# per-scene colour balance: yes = pale surfaces of each scene to beige, no = off
SCENE_BALANCE = {"yes": True, "true": True, "no": False, "false": False}.get(cfg.get("scene_balance", "no").strip().lower())
if SCENE_BALANCE is None:
    raise ValueError("scene_balance must be yes or no")
# natural: frames with less contrast than usual get less correction (open water has nothing to
# restore): none below the first share of the usual contrast, full from the second
CONTRAST_NONE, CONTRAST_FULL = (float(x) for x in cfg.get("contrast_fade", "0.45, 0.8").split(","))
# natural: correction fades out with height above the seabed: full up to the first (m), none from
# the second; empty = off
_alt_fade = cfg.get("altitude_fade", "").strip()
ALT_FULL, ALT_NONE = (float(x) for x in _alt_fade.split(",")) if _alt_fade else (0.0, 0.0)
# deflicker: each frame's average colour (per channel) is brought to its running average first
DEFLICKER = cfg.getboolean("deflicker", False)
# particle_method = blur: the thin near-vertical streaks found in each frame, before the correction
PARTICLE_BRIGHT = 3.0  # blur: a streak stands out at least this much (0-255)
PARTICLE_RATIO = 3.0  # blur: ... and this many times more than the seabed texture around it
PARTICLE_WORK = 0.5  # blur: streaks are found on a copy this size (faster); 1 = full size
FLICKER_EMA = 0.05  # share of each new frame in the running colour (~0.7 s at 30 fps)
FLICKER_MAX = 1.25  # at most this change of brightness per frame
SCENE_MAX = max(cfg.getfloat("scene_max", 1.3), 1.0)  # scene_balance: at most this change per channel
SCENE_EMA = 0.03  # scene_balance: share of each new frame in the running balance (~1 s at 30 fps)
if HAZE_FIT not in ("darkest", "colour"):
    raise ValueError("haze_fit must be darkest or colour")
START = cfg.get("start", "").strip()  # part of the video to enhance, e.g. 0:10
END = cfg.get("end", "").strip()  # empty = to the end
OUTSIDE = cfg.get("outside", "cut").strip().lower()  # cut or keep (unchanged) the rest
FIT_ON = cfg.get("fit_on", "part").strip().lower()  # fit the water on the whole video or the part
RANGE = cfg.get("range", "srt").strip().lower()  # srt (flat seabed) or depth_anything
DEPTH_MODEL = cfg.get("depth_model", "depth-anything/Depth-Anything-V2-Small-hf").strip()
# depth_anything: how its output becomes metres. relief = each row from the
# flat-seabed geometry, Depth Anything adds up to +-30 % within the row; curve = one smooth rising
# curve per frame from Depth Anything's values to metres, fitted so every row matches the geometry
DEPTH_FIT = cfg.get("depth_fit", "relief").strip().lower()
if DEPTH_FIT not in ("relief", "curve"):
    raise ValueError("depth_fit must be relief or curve")

MAX_RANGE = 40.0  # metres; rays that never reach the seabed (open water) count as this far
MIN_TRANSMISSION = 0.2  # caps the gain (5x) so far, dark areas don't turn into noise
CALIB_HEIGHT = 256  # calibration frames are shrunk to this height
DEPTH_SIZE = 518  # longest side the depth model works at
DEPTH_EMA = 0.3  # share of each new frame in the running depth map (less flicker)
# natural: bright sand / pale rock is balanced to light beige (Lab a, b) and set to
# SAND_BRIGHTNESS (sand in daylight is about L 75-82). Before, the brightest 0.5 % went to full
# white, which made the sand nearly white (L 91-93) and colourless.
NATURAL_SAND_AB = (1.0, 10.0)  # light beige; warmer (b 15+) turns seagrass khaki
NATURAL_SHADOW_AB = (-4.0, 8.0)  # natural, shadow_balance: Lab a, b of the darkest areas (dark olive)
_sand_srgb = cv2.cvtColor(np.array([[[max(SAND_BRIGHTNESS, 50), *NATURAL_SAND_AB]]], np.float32), cv2.COLOR_Lab2RGB)[0, 0]
NATURAL_SAND_LIN = np.where(_sand_srgb <= 0.04045, _sand_srgb / 12.92, ((_sand_srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
NATURAL_SAND = NATURAL_SAND_LIN / NATURAL_SAND_LIN.max()
NATURAL_MAX_GAIN = 3.0  # brightness boost for far areas is capped at this
NATURAL_HAZE = cfg.getfloat("haze", 1.0)  # natural: share of the (lowered) haze curve that is removed
LOOK_KEEP = 0.25  # natural: camera colour look this far from usual (log) is still corrected fully
LOOK_DROP = 0.45  # ... and from this far on not at all (stays near the original)
HAZE_BANDS = 8  # natural: vertical strips in which the haze is measured across the frame
HAZE_SHARE = 0.05  # natural: each frame's haze is followed on its darkest 5 % (0.5 % let a small dark object decide)
HAZE_ACROSS_MAX = 1.25  # natural: the haze may differ at most this much from one side of the frame to the middle
NATURAL_MAX_WB = max(cfg.getfloat("white_balance_limit", 3.5), 1.0)  # natural: the white balance may give one channel at most this times another
NATURAL_SMOOTH = 0.015  # natural: how closely the curves follow the points (log units, ~1.5 %)

if RANGE not in ("srt", "depth_anything"):
    raise ValueError("range must be srt or depth_anything")
DEPTH_ANYTHING = RANGE == "depth_anything"
NAME = METHOD + ("_depth" if DEPTH_ANYTHING else "")  # for file names
if NEIGHBOURS:  # particle removal, e.g. natural_pmedian5, natural_pbright5
    NAME += f"_p{PARTICLE_METHOD}{DENOISE_FRAMES // 2 * 2 + 1}"
elif PARTICLE_METHOD == "blur":
    NAME += "_pblur"
if LOCAL_CONTRAST > 0:
    NAME += f"_lc{LOCAL_CONTRAST:g}"  # e.g. natural_dn5_lc2

# e.g. input/GX010287_enhanced_natural_depth.mp4 (+ _depth with range = depth_anything)
# output_dir set: the video, its .srt and the preview / fit files all go to that folder instead
OUTPUT_DIR = cfg.get("output_dir", "").strip()
OUTPUT = (Path(OUTPUT_DIR) if OUTPUT_DIR else VIDEO.parent) / f"{VIDEO.stem}_enhanced_{NAME}.mp4"
RESULTS = Path(OUTPUT_DIR) if OUTPUT_DIR else Path("results") / "enhancement"
RESULTS.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video:  {VIDEO} ({total} frames, {total / fps / 60:.1f} min)")


def seconds(text):
    """'90', '1:30' or '0:01:30' -> seconds"""
    value = 0.0
    for part in text.split(":"):
        value = value * 60 + float(part)
    return value


if FIT_ON not in ("all", "part"):
    raise ValueError("fit_on must be all or part")
if OUTSIDE not in ("cut", "keep"):
    raise ValueError("outside must be cut or keep")
first = int(seconds(START) * fps) if START else 0  # frames first .. last-1 are enhanced
last = min(total, int(seconds(END) * fps)) if END else total
if not 0 <= first < last:
    raise ValueError(f"start ({START}) must be before end ({END}) and inside the video")
if first > 0 or last < total:
    print(f"Part:   {first / fps:.1f} - {last / fps:.1f} s enhanced, the rest is "
          + ("cut" if OUTSIDE == "cut" else "kept unchanged"))
print(f"Method: {METHOD}, camera tilt {TILT:g} deg, vertical FOV {VFOV:g} deg"
      + (f", range from {DEPTH_MODEL}" if DEPTH_ANYTHING else ""))


# ==========================================
# 1. ALTITUDE (RANGE) AND DEPTH FROM THE SRT FILE
# ==========================================

if SRT.exists():
    srt_entries = read_srt(SRT)  # timed by the logged clock when the subtitle times are wrong
    print(f"SRT:    {describe(srt_entries, SRT)}")
    times = np.array([(e["start"] + e["end"]) / 2 for e in srt_entries])
    depths = np.array([e["depth"] for e in srt_entries])
    alts = np.array([e["alt"] for e in srt_entries])
    # altimeter readings jump now and then; a running median over ~2.5 s steadies them
    half = max(1, int(round(1.25 * (len(times) - 1) / max(times[-1] - times[0], 1e-6))))
    alts = np.array([np.nanmedian(alts[max(0, i - half):i + half + 1]) for i in range(len(alts))])
    ok_alt, ok_depth = ~np.isnan(alts), ~np.isnan(depths)
    if not ok_alt.any():
        raise RuntimeError(f"No alt. values in {SRT}")
    print(f"        altitude {np.nanmin(alts):.1f}-{np.nanmax(alts):.1f} m, "
          f"depth {np.nanmin(depths):.1f}-{np.nanmax(depths):.1f} m")

    def altitude_at(t):
        return float(np.interp(t, times[ok_alt], alts[ok_alt]))

    def depth_at(t):
        return float(np.interp(t, times[ok_depth], depths[ok_depth])) if ok_depth.any() else np.nan
else:
    print(f"SRT:    none ({SRT} not found), using a fixed altitude of {ALTITUDE:g} m")
    DEPTH_CORRECTION = False

    def altitude_at(t):
        return ALTITUDE

    def depth_at(t):
        return np.nan


def row_range(altitude, height):
    """Distance from the camera to the seabed for each image row (flat seabed)."""
    # angle of each row's ray below the horizontal (pinhole camera, as in steps 4 and 6):
    # top edge = tilt - vfov/2, bottom edge = tilt + vfov/2
    y = ((np.arange(height) + 0.5) / height * 2 - 1) * np.tan(np.radians(VFOV) / 2)
    angle = np.radians(TILT) + np.arctan(y)
    with np.errstate(divide="ignore"):
        d = altitude / np.sin(angle)
    d[(angle <= np.radians(1)) | (d > MAX_RANGE)] = MAX_RANGE  # looking at open water
    return d.astype(np.float32)


# Depth Anything V2 (Yang et al. 2024) estimates the depth of every pixel from the image
# alone, so slopes, rocks and open water get their own range instead of a flat seabed.
# It only gives relative depth (near / far), so it is scaled to metres with the altimeter:
# the bottom rows of the frame (looking almost straight down, where the flat-seabed range
# is reliable) are set to their range from alt.

if DEPTH_ANYTHING:
    import torch
    from transformers import AutoModelForDepthEstimation

    depth_device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(depth_device).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=depth_device).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=depth_device).view(1, 3, 1, 1)
    depth_state = {"range": None}
    print(f"Depth model on {depth_device}: {DEPTH_MODEL}")

    def pixel_range(frame, altitude, reset=False):
        """Distance in metres from the camera to every pixel, at the depth model's size."""
        h, w = frame.shape[:2]
        scale = DEPTH_SIZE / max(h, w)
        size = (max(14, round(w * scale / 14) * 14), max(14, round(h * scale / 14) * 14))
        rgb = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)[..., ::-1]
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(depth_device).permute(2, 0, 1)[None].float() / 255
        with torch.inference_mode():
            disparity = depth_net(pixel_values=(x - MEAN) / STD).predicted_depth[0].float().cpu().numpy()
        disparity = cv2.resize(disparity, size)  # bigger = closer
        rows = disparity.shape[0]
        bottom = slice(int(rows * 0.9), rows)
        k = np.median(disparity[bottom]) * np.median(row_range(altitude, rows)[bottom])
        with np.errstate(divide="ignore"):
            d = np.where(disparity > 1e-3, k / disparity, MAX_RANGE)
        # Depth Anything's far range is not reliable in this water: scaled at the bottom it put the
        # top of the frame at ~30 m, with scale and offset fitted at ~6 m, while the seabed's motion
        # in the image (optical flow) matches the flat-seabed geometry, ~10 m. (Tested on GX010287:
        # Small, Base and Large all put the top at the 40 m cap; scaled like this, natural got a
        # yellow, too bright top.) So each row's distance comes from the geometry (tilt, vfov,
        # altitude), and Depth Anything only adds the relief within the row (rocks, clumps), +-30 %.
        plane = row_range(altitude, rows)
        if DEPTH_FIT == "curve":
            d = curve_range(disparity, plane)
        else:
            d = relief_range(d, plane)
        d = np.clip(d, 0.3, MAX_RANGE).astype(np.float32)
        last = depth_state["range"]
        if not reset and last is not None and last.shape == d.shape:
            d = DEPTH_EMA * d + (1 - DEPTH_EMA) * last
        depth_state["range"] = d
        return d

    def relief_range(d, plane):
        """depth_fit = relief: each row's distance from the geometry, Depth Anything's relief within it"""
        rows = d.shape[0]
        cols = slice(d.shape[1] // 4, 3 * d.shape[1] // 4)
        # each row's reference, smoothed from top to bottom: the plain median of the middle
        # columns flips between rock and sand from one row to the next (straight horizontal steps)
        log_ref = np.log(np.maximum(np.median(d[:, cols], axis=1), 1e-3))
        log_ref = cv2.GaussianBlur(log_ref[:, None].astype(np.float32), (1, 0), sigmaX=0,
                                   sigmaY=max(rows / 40, 1), borderType=cv2.BORDER_REFLECT)
        # soft limit to +-30 % (tanh) instead of a hard clip, which made flat patches with sharp edges
        limit = np.log(1.3)
        log_rel = np.log(np.maximum(d, 1e-3)) - log_ref
        relief = np.exp(limit * np.tanh(log_rel / limit))
        return np.where(plane[:, None] < MAX_RANGE, plane[:, None] * relief, MAX_RANGE)

    def _rising(y):
        """pool-adjacent-violators: the closest non-decreasing sequence to y"""
        vals, counts = [], []
        for v in y.astype(np.float64):
            vals.append(v)
            counts.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                n = counts[-2] + counts[-1]
                vals[-2] = (vals[-2] * counts[-2] + vals[-1] * counts[-1]) / n
                counts[-2] = n
                del vals[-1], counts[-1]
        return np.repeat(vals, counts)

    def curve_range(disparity, plane):
        """depth_fit = curve: one rising curve per frame from Depth Anything's value (bigger =
        nearer) to 1 / distance, fitted on the rows so each row's median matches the geometry.
        Every pixel keeps Depth Anything's near / far order; the relief comes out of the curve."""
        seabed = plane < MAX_RANGE
        x = np.median(disparity, axis=1)[seabed]
        if len(x) < 4:
            return np.repeat(plane[:, None], disparity.shape[1], axis=1)
        order = np.argsort(x)
        x, y = x[order], _rising(1.0 / plane[seabed][order])
        knots_x, start = np.unique(np.round(x, 5), return_index=True)
        ends = list(start[1:]) + [len(y)]
        knots_y = np.array([y[a:b].mean() for a, b in zip(start, ends)])
        if len(knots_x) < 2:
            return np.repeat(plane[:, None], disparity.shape[1], axis=1)
        # straight on beyond the fitted values, with the slope of the outer fifth of the knots
        k = max(2, len(knots_x) // 5)
        slope_lo = (knots_y[k - 1] - knots_y[0]) / max(knots_x[k - 1] - knots_x[0], 1e-6)
        slope_hi = (knots_y[-1] - knots_y[-k]) / max(knots_x[-1] - knots_x[-k], 1e-6)
        inv = np.interp(disparity, knots_x, knots_y)
        inv = np.where(disparity < knots_x[0], knots_y[0] + (disparity - knots_x[0]) * slope_lo, inv)
        inv = np.where(disparity > knots_x[-1], knots_y[-1] + (disparity - knots_x[-1]) * slope_hi, inv)
        return 1.0 / np.maximum(inv, 1.0 / MAX_RANGE)


# ==========================================
# 2. COLOUR HELPERS (the correction works on linear light, not on gamma-encoded pixels)
# ==========================================

_v = np.arange(256) / 255.0
TO_LINEAR = np.where(_v <= 0.04045, _v / 12.92, ((_v + 0.055) / 1.055) ** 2.4).astype(np.float32)
_v = np.linspace(0, 1, 4096)
TO_SRGB = (255 * np.where(_v <= 0.0031308, 12.92 * _v, 1.055 * _v ** (1 / 2.4) - 0.055) + 0.5).astype(np.uint8)


def linear(frame_bgr):
    return TO_LINEAR[frame_bgr[..., ::-1]]  # RGB, 0-1 linear


def encode(rgb_linear):
    idx = (np.clip(rgb_linear, 0, 1) * 4095).astype(np.uint16)
    return TO_SRGB[idx][..., ::-1]  # back to BGR uint8


# The per-pixel correction (linear -> gain and haze -> sRGB) is most of the time per frame;
# with an NVIDIA GPU it runs there in PyTorch (same maths, ~10x faster than numpy).
try:
    import torch
    GPU = torch.device("cuda") if torch.cuda.is_available() else None
except ImportError:
    GPU = None
if GPU is not None:
    TO_LINEAR_GPU = torch.from_numpy(TO_LINEAR).to(GPU)
    TO_SRGB_GPU = torch.from_numpy(TO_SRGB).to(GPU)


def on_device(a):
    """numpy array -> float32 tensor on the GPU (unchanged without a GPU)"""
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(GPU) if GPU is not None else a


def apply_correction(frame, gain, offset, amount=1.0):
    """encode(linear(frame) * gain + offset), blended with the frame by `amount`.
    gain and offset are RGB and broadcast to (h, w, 3); from on_device()."""
    if GPU is None:
        out = encode(linear(frame) * gain + offset)
        return cv2.addWeighted(out, amount, frame, 1 - amount, 0) if amount < 1 else out
    x = torch.from_numpy(frame).to(GPU)  # BGR, so gain and offset are flipped instead of the frame
    out = TO_LINEAR_GPU[x.long()] * gain.flip(-1) + offset.flip(-1)
    out = TO_SRGB_GPU[(out.clamp(0, 1) * 4095).long()]
    if amount < 1:
        out = (out.float() * amount + x.float() * (1 - amount)).round().clamp(0, 255).to(torch.uint8)
    return out.cpu().numpy()


def fit_size(h, w, max_side):
    scale = min(1.0, max_side / max(h, w)) if max_side > 0 else 1.0
    return int(round(w * scale)) // 2 * 2, int(round(h * scale)) // 2 * 2


# ==========================================
# 3. LEARN THE WATER FROM SAMPLE FRAMES
# ==========================================
# Image formation (Akkaynak & Treibitz, Sea-thru 2019), per colour channel c:
#     I_c = J_c * exp(-beta_D_c * d)  +  B_c(d)
# I = what the camera sees, J = true colour, d = range to the scene,
# exp(-beta_D * d) = how much of the colour survives the water, B(d) = backscatter (haze).
# The water parameters are fitted once for the whole video, so the colours don't flicker.

def spread(a, b, n):
    """n frame numbers spread over frames a .. b"""
    return np.unique(np.linspace(a + 0.02 * (b - a), a + 0.98 * (b - a), n).astype(int))


# fit_on = all: the water is fitted on the whole video (more varied scenes, so the content of
# one short part doesn't bias it); part: only on the start - end part that is enhanced
picks = spread(0, total, CALIB_FRAMES) if FIT_ON == "all" else spread(first, last, CALIB_FRAMES)
preview_picks = spread(first, last, 20)  # the preview always shows the enhanced part
print(f"Fitting the water on {CALIB_FRAMES} sample frames...")
samples = []  # (linear RGB, range per pixel, camera depth, frame index)
for index in picks:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
        continue
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (round(w * CALIB_HEIGHT / h), CALIB_HEIGHT), interpolation=cv2.INTER_AREA)
    if DEPTH_ANYTHING:
        rng = cv2.resize(pixel_range(frame, altitude_at(index / fps), reset=True), small.shape[1::-1])
    else:
        rng = np.repeat(row_range(altitude_at(index / fps), CALIB_HEIGHT)[:, None], small.shape[1], axis=1)
    samples.append((linear(small), rng, depth_at(index / fps), int(index)))

if len(samples) < 5:
    raise RuntimeError("Could not read enough frames from the video")

# Camera white balance. The GoPro's automatic white balance switches
# in some stretches of a dive (on GX010175 over bright sand and in open water: the whole
# frame turns lavender). One water correction cannot fit both looks, and applied to those
# stretches it turns them pink. The camera's colour look is measured per frame (average
# colour of the whole frame, colour only); frames far from the dive's usual look are left
# out of the water fit, and the correction fades out on them (they stay near the original).
def colour_look(lin):
    m = np.log(np.maximum(lin.reshape(-1, 3).mean(0), 1e-5))
    return m - m.mean()


usual_look = np.median([colour_look(s[0]) for s in samples], axis=0)


def look_distance(lin):
    return float(np.abs(colour_look(lin) - usual_look).max())


far = np.array([look_distance(s[0]) > LOOK_KEEP for s in samples])
if far.any() and (~far).sum() >= 10:
    print(f"Camera colour: {int(far.sum())} of {len(samples)} sample frames have a different camera "
          f"white balance; left out of the fit, corrected less in the video")
    samples = [s for s, f in zip(samples, far) if not f]
    usual_look = np.median([colour_look(s[0]) for s in samples], axis=0)

pixels = np.concatenate([s[0].reshape(-1, 3) for s in samples])
ranges = np.concatenate([s[1].ravel() for s in samples])

# --- backscatter: the darkest pixels at each range are almost pure haze (J ~ 0) ---
edges = np.unique(np.quantile(ranges, np.linspace(0, 1, 16)))
dark_d, dark_rgb = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    inside = np.flatnonzero((ranges >= lo) & (ranges <= hi))
    if len(inside) < 200:
        continue
    brightness = pixels[inside].sum(1)
    darkest = inside[brightness <= np.quantile(brightness, 0.01)]
    dark_d.append(ranges[darkest])
    dark_rgb.append(pixels[darkest])
dark_d, dark_rgb = np.concatenate(dark_d), np.concatenate(dark_rgb)


def backscatter_simple(d, b_inf, beta_b):
    return b_inf * (1 - np.exp(-beta_b * d))


def backscatter_seathru(d, b_inf, beta_b, j_res, beta_res):
    # Sea-thru eq. 10: the extra term absorbs the faint scene colour left in the dark pixels
    return b_inf * (1 - np.exp(-beta_b * d)) + j_res * np.exp(-beta_res * d)


back_params = []
for c in range(3):
    try:
        # soft_l1 = robust fit: single stray dark pixels pull much less than in least squares
        p, _ = curve_fit(backscatter_seathru, dark_d, dark_rgb[:, c], p0=[0.5, 0.5, 0.01, 0.5],
                         bounds=([0, 0, 0, 0], [1, 5, 1, 5]), loss="soft_l1", f_scale=0.02,
                         maxfev=20000)
    except RuntimeError:
        p = [float(np.quantile(dark_rgb[:, c], 0.5)), 1.0, 0.0, 1.0]
    back_params.append([float(x) for x in p])

if HAZE_FIT == "colour":
    # Haze is light scattered by the water itself: the same colour at every range, only its
    # amount grows with range. Its colour is the plateau the darkest pixels reach far away
    # (b_inf above). Near by, the darkest pixels are dark seagrass, whose own green the
    # per-channel fit took for haze. So the amount at each range comes from the dark channel
    # instead: each dark pixel divided by the haze colour, the lowest of the three (the
    # channel the scene adds least to), and one curve for all channels.
    b_inf = np.array([back_params[c][0] for c in range(3)])
    dark_ratio = (dark_rgb / np.maximum(b_inf, 1e-6)).min(1)

    def amount(d, a, beta):
        return a * (1 - np.exp(-beta * d))

    try:
        (a_fit, beta_fit), _ = curve_fit(amount, dark_d, dark_ratio, p0=[1.0, 0.3], bounds=([0, 0], [1.5, 5]),
                                         loss="soft_l1", f_scale=0.05, maxfev=20000)
    except RuntimeError:
        a_fit, beta_fit = 1.0, float(np.median([p[1] for p in back_params]))
    back_params = [[float(a_fit * b_inf[c]), float(beta_fit), 0.0, 1.0] for c in range(3)]
    print(f"Haze from the water colour: R/G/B {np.round(b_inf * a_fit, 3)}, one rise rate {beta_fit:.2f} /m")


def backscatter(d):
    """Haze colour for each range d, shape (*d.shape, 3)."""
    # only the haze itself: the fitted J' term is leftover scene colour in the dark
    # pixels (e.g. the green of shaded seagrass), not haze, so it is not removed
    return (np.stack([backscatter_simple(d, *back_params[c][:2]) for c in range(3)], axis=-1)
            * back_scale).astype(np.float32)


back_scale = np.ones(3, np.float32)
# The haze curve runs through the middle of the darkest pixels, so subtracting it
# pushes half of them below zero: dark seagrass loses its green and turns grey /
# purple. As in DeepSeeColor, the haze may not exceed what the image allows: it is
# lowered until at most 0.5 % of the pixels at any range would go below zero.
ratio = pixels / np.maximum(backscatter(ranges), 1e-6)
edges_r = np.unique(np.quantile(ranges, np.linspace(0, 1, 16)))
for lo, hi in zip(edges_r[:-1], edges_r[1:]):
    inside = (ranges >= lo) & (ranges <= hi)
    if inside.sum() >= 200:
        back_scale = np.minimum(back_scale, np.quantile(ratio[inside], 0.005, axis=0))
# one factor for all channels: the haze colour is a property of the water and stays as
# fitted; lowering each channel on its own would force the darkest pixels to black in
# every channel and strip the green from shaded seagrass
back_scale = np.full(3, np.clip(back_scale.min(), 0.3, 1) * NATURAL_HAZE, np.float32)
print(f"Haze lowered to {back_scale[0]:.2f} of the fitted curve")


def visibility(lin):
    return float(lin.mean(2).std())


def haze_envelope(lin, d, share=0.005):
    """per channel: how far the haze curve could be raised before `share` of the pixels go
    below zero"""
    ratio = (lin / np.maximum(backscatter(d), 1e-6)).reshape(-1, 3)
    return np.quantile(ratio, share, axis=0)


# each frame's haze is followed relative to its typical value: the average haze colour
# stays as fitted (keeps the green in shaded seagrass), while a frame with a little
# more or less haze, or a slightly different haze colour, is cleaned accordingly, so
# the colour stays steady from frame to frame
typical_envelope = np.median([haze_envelope(s[0], s[1], HAZE_SHARE) for s in samples], axis=0)
# (the haze across each frame, from the strips, is added in frame_haze below)
# how much there is to see: brightness contrast of a frame. Frames with much less than
# usual (open water, descent) have almost no colour left to recover, and forcing a
# correction there only invents colours, so it fades out on them
typical_visibility = float(np.median([visibility(s[0]) for s in samples]))


def frame_haze(lin, d):
    """this frame's haze per channel, relative to the fitted (lowered) curve, across the
    frame: (HAZE_BANDS, 3), one row per vertical strip from left to right.
    The level and colour come from the whole frame, as above. Light from one side (the
    sun's direction) can make the haze thicker on that side; measured in each strip and
    smoothed into a curve across the width, it is removed there too, instead of being
    left in and amplified into a colour cast down one side."""
    level = np.clip(haze_envelope(lin, d, HAZE_SHARE) / typical_envelope, 0.5, 1.5)
    strips = np.array_split(np.arange(lin.shape[1]), HAZE_BANDS)
    # per strip (as the level) from its darkest HAZE_SHARE: a small dark object in one strip
    # (part of the vehicle, a shadow) otherwise makes that strip look haze-free and pushes the
    # haze onto the others (GX010287 0:58: a red band down the frame)
    across = np.log(np.maximum([haze_envelope(lin[:, s], d[:, s], HAZE_SHARE) for s in strips], 1e-6))
    x = np.linspace(-1, 1, HAZE_BANDS)
    shape = np.stack([np.polyval(np.polyfit(x, across[:, c], 2), x) for c in range(3)], axis=1)
    shape -= shape.mean(0)  # only the change across the frame; the level stays as above
    # light from one side changes the haze across the frame a little, never several times
    shape = np.clip(shape, -np.log(HAZE_ACROSS_MAX), np.log(HAZE_ACROSS_MAX))
    return np.clip(level * np.exp(shape), 0.5, 1.5).astype(np.float32)


def haze_columns(field, width):
    """(HAZE_BANDS, 3) haze across the frame -> (width, 3), one value per pixel column"""
    x = (np.arange(HAZE_BANDS) + 0.5) / HAZE_BANDS * width
    cols = np.arange(width) + 0.5
    return np.stack([np.interp(cols, x, field[:, c]) for c in range(3)], axis=1).astype(np.float32)


# --- how the scene colour fades with range ---
# The seabed has on average the same colour near and far, so the change of the direct
# signal (I - B) with range shows what the water does to it.
direct = [np.clip(s[0] - backscatter(s[1]) * haze_columns(frame_haze(s[0], s[1]), s[0].shape[1]), 0, None)
          for s in samples]
signal = np.concatenate([s.reshape(-1, 3) for s in direct])

near = ranges < MAX_RANGE
edges = np.unique(np.quantile(ranges[near], np.linspace(0, 1, 21)))
bin_d, bin_log = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    inside = near & (ranges >= lo) & (ranges <= hi)
    if inside.sum() < 200:
        continue
    # mean without the darkest / brightest 5 % (shadows, glints)
    values = signal[inside]
    lum = values.mean(1)
    lo_l, hi_l = np.quantile(lum, [0.05, 0.95])
    level = values[(lum >= lo_l) & (lum <= hi_l)].mean(0)
    if (level > 1e-4).all():
        bin_d.append(float(np.median(ranges[inside])))
        bin_log.append(np.log(level))
bin_d, bin_log = np.array(bin_d), np.array(bin_log)
if len(bin_d) < 3:
    raise RuntimeError("Not enough range variation to fit the attenuation; check tilt / vfov / altitude")


def smooth_curve(d, y):
    """A smooth curve through the points (smoothing spline over log range, so the many
    near points and the few far ones count alike). Only points far off it (> 4 robust
    standard deviations) are left out, then it is fitted again."""
    x = np.log(d)
    keep = np.ones(len(d), bool)
    for _ in range(2):
        spline = UnivariateSpline(x[keep], y[keep], k=3, s=keep.sum() * NATURAL_SMOOTH ** 2)
        residual = y - spline(x)
        spread_r = max(1.4826 * np.median(np.abs(residual[keep])), NATURAL_SMOOTH)
        good = np.abs(residual) <= 4 * spread_r
        if good[keep].all():
            break
        keep &= good
    return spline, keep


# natural: a smooth curve per channel through the average signal at each range
# (no physics formula needed here: it only has to describe how the scene looks at
# each range, so it can be evened out). From the curves:
#  1. brightness: the average of the channels; far areas are brightened to match the
#     brightest range (the same gain for all channels, so no colour cast; capped), and
#  2. colour: each channel's share of the brightness. Every range is given the colour
#     balance seen nearest the camera, so near and far look alike (no red / purple tops).
curves, kept = [], []
for c in range(3):
    spline, keep = smooth_curve(bin_d, bin_log[:, c])
    curves.append(spline)
    kept.append(keep)
kept = np.array(kept).T
fit_lo, fit_hi = float(bin_d.min()), float(bin_d.max())
d_ref = fit_lo  # the colour nearest the camera is the reference
x_grid = np.log(np.linspace(fit_lo, fit_hi, 200))
lum_max = float(np.mean([s(x_grid) for s in curves], axis=0).max())
atten_params = {"spline_points_m": bin_d.tolist(), "smooth": NATURAL_SMOOTH, "reference_m": d_ref}


def predicted_log(d):
    """the fitted curves, held constant outside the measured ranges"""
    x = np.log(np.clip(d, fit_lo, fit_hi))
    return np.stack([s(x) for s in curves], axis=-1)


def restore_gain(d):
    """brightness gain (capped) x colour correction to the near-range balance"""
    curve = predicted_log(np.asarray(d, np.float64))
    lum = curve.mean(-1, keepdims=True)
    ref = predicted_log(np.array([d_ref]))[0]
    g = np.minimum(np.exp(lum_max - lum), NATURAL_MAX_GAIN)
    shift = (curve - lum) - (ref - ref.mean())
    if FAR_COLOUR_LIMIT > 1:
        # the colour change with range assumes the seabed looks alike near and far; when
        # the far part is often a different bottom (meadow vs pale rock) it over-corrects,
        # e.g. an orange top. Limited to this factor per channel.
        shift = np.clip(shift, -np.log(FAR_COLOUR_LIMIT), np.log(FAR_COLOUR_LIMIT))
    return (g * np.exp(-shift)).astype(np.float32)


def gain_curve(d):
    return restore_gain(d)


# --- camera depth: deeper water lights the scene with bluer, dimmer daylight ---
restored = [D * restore_gain(s[1]) for D, s in zip(direct, samples)]
frame_depth = np.array([s[2] for s in samples])
depth_slope = np.zeros(3)
depth_ref = float(np.nanmedian(frame_depth)) if DEPTH_CORRECTION else 0.0
if DEPTH_CORRECTION and np.nanstd(frame_depth) > 0.3:
    valid = ~np.isnan(frame_depth)
    means = np.array([J[s[1] < MAX_RANGE].mean(0) for J, s in zip(restored, samples)])
    for c in range(3):
        # log(colour) ~ -K_d * depth  (daylight fades exponentially with depth)
        slope = np.polyfit(frame_depth[valid], np.log(np.maximum(means[valid, c], 1e-5)), 1)[0]
        depth_slope[c] = float(np.clip(slope, -0.3, 0.0))
else:
    DEPTH_CORRECTION = False


def depth_gain(depth):
    if not DEPTH_CORRECTION or np.isnan(depth):
        return np.ones(3, np.float32)
    return np.exp(-depth_slope * (depth - depth_ref)).astype(np.float32)


# --- colour across the frame (natural) ---
# Light can reach the scene greener on one side than on the other (the sun's direction,
# the housing): the same colour gradient from left to right in every frame, whatever the
# scene. It is measured as the median, over all sample frames, of each strip's colour
# (not brightness) relative to its frame, smoothed into a curve, and evened out by a
# colour gain per column.
x = np.linspace(-1, 1, HAZE_BANDS)
profiles = []
for J in restored:
    strips = np.array_split(np.arange(J.shape[1]), HAZE_BANDS)
    m = np.log(np.maximum([J[:, s].reshape(-1, 3).mean(0) for s in strips], 1e-6))
    chroma = m - m.mean(1, keepdims=True)  # colour only
    profiles.append(chroma - chroma.mean(0))  # its change across the frame
profile = np.median(profiles, axis=0)
profile = np.stack([np.polyval(np.polyfit(x, profile[:, c], 2), x) for c in range(3)], axis=1)
colour_across = np.clip(np.exp(-profile), 0.5, 2).astype(np.float32)
restored = [J * haze_columns(colour_across, J.shape[1])[None] for J in restored]

# --- white balance + contrast ---
everything = np.concatenate([(J * depth_gain(s[2])).reshape(-1, 3) for J, s in zip(restored, samples)])
# the brightest surfaces (sand, pale rock: top 2-8 % brightness, not the glints above)
# are set to light beige; then one stretch for all channels, so no cast is added
lum = everything.mean(1)
lo_l, hi_l = np.quantile(lum, [0.92, 0.98])
sand = everything[(lum >= lo_l) & (lum <= hi_l)].mean(0)
white = (NATURAL_SAND / np.maximum(sand, 1e-5)).astype(np.float32)
# murky water: when the sand asks for a very strong correction (red x2, green x0.4),
# little real colour is left to recover, and forcing it turns noise and floating
# particles pink; the balance is then limited so that no channel gets more than
# NATURAL_MAX_WB times the gain of another (the water keeps a little of its tint)
spread_w = float(np.log(white.max() / white.min()))
if spread_w > np.log(NATURAL_MAX_WB):
    lw = np.log(white)
    white = np.exp(lw.mean() + (lw - lw.mean()) * np.log(NATURAL_MAX_WB) / spread_w).astype(np.float32)
    print(f"White balance limited: sand asked for {np.exp(spread_w):.1f}x between channels, "
          f"capped at {NATURAL_MAX_WB:g}x")
white /= white.mean()
everything = everything * white
low = np.float32(np.quantile(everything, 0.005))
if SAND_BRIGHTNESS > 0:
    # one stretch for all channels that puts the sand at SAND_BRIGHTNESS; the few
    # things brighter than sand (glints, white shells) may clip
    # the brightest surfaces (97-99.5 %: sand, pale rock) are what gets that brightness
    lum_wb = everything.mean(1)
    b_lo, b_hi = np.quantile(lum_wb, [0.97, 0.995])
    sand_now = float(everything[(lum_wb >= b_lo) & (lum_wb <= b_hi)].mean())
    span = np.float32(max((sand_now - low) / float(NATURAL_SAND_LIN.mean()), 1e-4))
else:
    span = np.float32(max(np.quantile(everything, 0.995) - low, 1e-4))
_restore = restore_gain


def restore_gain(d):
    return _restore(d) * white


# --- shadow balance (natural) ---
# The haze is fitted on the darkest pixels, taken to be black. In a dense Posidonia meadow
# they are dark green seagrass, so the "haze" holds some of its green, and removing it
# leaves the shadows and the seagrass mauve / pink (the white balance for the sand cuts
# green further). Less haze is removed in the channels the shadows lack, until the darkest
# 20 % of the sample frames come out dark olive (NATURAL_SHADOW_AB) instead. Bright
# surfaces hardly change: the haze is a small part of their light.
shadow_k = np.ones(3, np.float32)  # share of the haze removed, per channel
if SHADOW_BALANCE:
    terms = []  # per sample pixel: corrected value with all haze removed, and the haze term
    for s in samples:
        lin, d, depth = s[0], s[1], s[2]
        w_s = lin.shape[1]
        g = restore_gain(d) * depth_gain(depth) * haze_columns(colour_across, w_s)[None] / span
        haze = backscatter(d) * haze_columns(frame_haze(lin, d), w_s)[None]
        terms.append(np.stack([(lin * g).reshape(-1, 3), (haze * g).reshape(-1, 3)]))
    terms = np.concatenate(terms, axis=1)
    signal, haze_term = terms[0] - low / span, terms[1]
    target_lab = None
    for _ in range(4):  # the clipping at 0 changes with the factors: a few rounds
        out = np.clip(signal - haze_term * shadow_k, 0, 1)
        lum = out.mean(1)
        dark = lum <= np.quantile(lum, 0.2)
        m = out[dark].mean(0)
        if target_lab is None:  # the shadows keep their brightness, get the olive colour
            m_srgb = TO_SRGB[(np.clip(m, 0, 1) * 4095).astype(int)].astype(np.float32) / 255
            L = float(cv2.cvtColor(m_srgb[None, None], cv2.COLOR_RGB2Lab)[0, 0, 0])
            target_lab = np.array([[[L, *NATURAL_SHADOW_AB]]], np.float32)
            t_srgb = np.clip(cv2.cvtColor(target_lab, cv2.COLOR_Lab2RGB)[0, 0], 0, 1)
            target = np.where(t_srgb <= 0.04045, t_srgb / 12.92, ((t_srgb + 0.055) / 1.055) ** 2.4)
        # only less haze is removed (never more): the target is scaled so that it needs
        # nothing taken away in any channel
        add = target * float(np.max(m / np.maximum(target, 1e-6))) - m
        h = haze_term[dark].mean(0)
        shadow_k = np.clip(shadow_k - add / np.maximum(h, 1e-6), 0.3, 1).astype(np.float32)
    print(f"Shadow balance: haze removed R/G/B x {shadow_k.round(2)} (darkest 20 % to dark olive)")


# ==========================================
# 4. PARTICLES, LOCAL CONTRAST AND DISTANCE COLOUR BALANCE
# ==========================================
# Particle removal with neighbouring frames (particle_method = median / bright): the camera sees
# the same seabed in several frames in a row. The frames around each one are aligned to it
# (dense optical flow) and the median of each pixel is taken. Noise drops, and what moves
# differently from the seabed (floating particles, also fish and drifting debris) is mostly
# removed. It runs before the colour correction, so the colour boost works on a cleaner picture.
# Local contrast: CLAHE (contrast-limited adaptive histogram equalisation) on the brightness
# only, so texture such as seagrass leaves stands out; the colours are left as they are.

DENOISE = DENOISE_FRAMES // 2 * 2 + 1 if DENOISE_FRAMES > 1 else 1  # odd: a centred window
if DENOISE > 1:
    import torch

    flow_engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    median_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Particle removal: {PARTICLE_METHOD}, {DENOISE} aligned frames"
          + (" (middle value everywhere)" if PARTICLE_METHOD == "median" else
             f" (only what is {PARTICLE_STEP:g} brighter than in them)"))

    def align(ref, other):
        """warp `other` onto `ref`; the flow is worked out at half size, then scaled up"""
        h, w = ref.shape[:2]
        a = cv2.cvtColor(cv2.resize(ref, (w // 2, h // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(other, (w // 2, h // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        flow = cv2.resize(flow_engine.calc(a, b, None), (w, h)) * 2
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return cv2.remap(other, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def denoise(frames, centre):
        """median of the frames aligned to frames[centre] (fewer at the ends of the video)"""
        if len(frames) < 3:
            return frames[centre]
        aligned = [f if k == centre else align(frames[centre], f) for k, f in enumerate(frames)]
        stack = torch.from_numpy(np.stack(aligned)).to(median_device).short()
        median = stack.median(dim=0).values
        if PARTICLE_METHOD == "bright":
            # only what is clearly brighter than in the neighbouring frames (floating particles,
            # lit by the lamps) is replaced; everything else keeps the frame's own detail
            frame = stack[centre]
            brighter = (frame - median).float().mean(-1) > PARTICLE_STEP
            brighter = torch.nn.functional.max_pool2d(brighter[None, None].float(), PARTICLE_GROW, 1,
                                                      PARTICLE_GROW // 2)[0, 0] > 0
            median = torch.where(brighter[..., None], median, frame)
        return median.to(torch.uint8).cpu().numpy()
else:
    def denoise(frames, centre):
        return frames[centre]
if PARTICLE_METHOD == "blur":
    print("Particle removal: blur (streaks filled with the colour around them)")


if LOCAL_CONTRAST > 0:
    print(f"Local contrast: CLAHE strength {LOCAL_CONTRAST:g}")
clahe = cv2.createCLAHE(clipLimit=LOCAL_CONTRAST, tileGridSize=(8, 8)) if LOCAL_CONTRAST > 0 else None


def local_contrast(bgr):
    if clahe is None:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Distance colour balance (range_balance, 0-1): after the correction the far part of the frame
# still differs from the near part (yellow / brown / green top, flat far rock), because the far
# light lost more red and gained more haze than any range model gives back. Here the pale,
# textured surfaces (sand, rock; not smooth open water) are measured from top to bottom, and each
# height is brought to the near seabed (bottom 25 %): colour shifted (at most 10 Lab units, 5 towards red),
# darkened where brighter (never brightened much: that turns the haze milky) and its fine detail
# raised (at most 1.8x). The curves are followed smoothly over time, so it does not flicker.
if RANGE_BALANCE > 0:
    print(f"Distance colour balance: strength {RANGE_BALANCE:g}")
RB_SAMPLES = 64  # heights at which the top-to-bottom curves are kept
rb_state = {"curves": None}


def _rb_fit(y, v, w, deg=2):
    ok = w > 0
    if ok.sum() < deg + 2:
        return None
    c = np.polyfit(y[ok], v[ok], deg, w=np.sqrt(w[ok]))
    return np.polyval(c, np.clip(np.linspace(0, 1, RB_SAMPLES), y[ok].min(), y[ok].max()))


def _rb_measure(bgr):
    small = cv2.resize(bgr, (max(bgr.shape[1] // 4, 8), max(bgr.shape[0] // 4, 8)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0]
    detail = L - cv2.GaussianBlur(L, (0, 0), 3)
    texture = cv2.GaussianBlur(np.abs(detail), (0, 0), 2)
    pale = (L >= np.percentile(L, 65)) & (texture >= np.percentile(texture, 35))
    h = L.shape[0]
    y = (np.arange(h) + 0.5) / h
    cnt = pale.sum(1).astype(np.float64)
    w = np.where(cnt >= 6, cnt, 0)
    curves = {k: _rb_fit(y, np.where(cnt > 0, (lab[..., i] * pale).sum(1) / np.maximum(cnt, 1), 0), w)
              for i, k in enumerate(("L", "a", "b"))}
    curves["detail"] = _rb_fit(y, np.sqrt((detail ** 2).mean(1)), np.ones(h))
    return None if any(v is None for v in curves.values()) else curves


def range_balance(bgr, reset=False):
    if RANGE_BALANCE <= 0:
        return bgr
    c = _rb_measure(bgr)
    if c is not None:
        last = rb_state["curves"]
        rb_state["curves"] = c if (reset or last is None) else {k: 0.15 * c[k] + 0.85 * last[k] for k in c}
    c = rb_state["curves"]
    if c is None:
        return bgr
    s, near = RANGE_BALANCE, slice(int(RB_SAMPLES * 0.75), RB_SAMPLES)
    # towards red at most 5: far sand measured greenish otherwise turned the top pink (~6 % of frames)
    da = np.clip(c["a"][near].mean() - c["a"], -10, 5) * s
    db = np.clip(c["b"][near].mean() - c["b"], -10, 10) * s
    gl = np.clip(c["L"][near].mean() / np.maximum(c["L"], 1), 0.85, 1.03) ** s
    gd = np.clip(c["detail"][near].mean() / np.maximum(c["detail"], 1e-3), 1.0, 1.8) ** s
    H = bgr.shape[0]
    grid = np.linspace(0, 1, RB_SAMPLES)
    at = lambda v: np.interp((np.arange(H) + 0.5) / H, grid, v).astype(np.float32)[:, None]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0]
    base = cv2.GaussianBlur(L, (0, 0), 12 * H / 1280)
    lab[..., 0] = (base + (L - base) * at(gd)) * at(gl)
    lab[..., 1] += at(da)
    lab[..., 2] += at(db)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


def read_window(index, size=None):
    """frame `index` and its DENOISE // 2 neighbours on each side (by seeking), denoised"""
    half = DENOISE // 2
    frames = []
    for k in range(max(0, index - half), min(total, index + half + 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA) if size else f)
    if not frames:
        return None
    return denoise(frames, min(index - max(0, index - half), len(frames) - 1))


# ==========================================
# 5. THE CORRECTION FOR ONE FRAME
# ==========================================
# With a flat seabed the range is the same along a row, so the correction is per row and channel:
#     out = I * gain[row] + offset[row]

# With Depth Anything the range differs per pixel: the same correction is worked out on the
# depth map's grid and then resized to the frame.

def correction(d, depth):
    """gain, and the haze to subtract (already multiplied by the gain), for ranges d"""
    g = restore_gain(d) * depth_gain(depth) / span
    return g, backscatter(d) * shadow_k * g


haze_state = {"scale": None, "amount": None}
HAZE_EMA = 0.1  # natural: share of each new frame in the running haze strength (~0.3 s)


flicker_state = {"level": None}  # deflicker: running average colour (RGB, linear)


def remove_particles(frame):
    """remove_particles: floating particles lit by the vehicle's lamps, filled in from around them.
    In the original they are faint whitish streaks, but the correction boosts red ~2x and turns them
    into red dots (GX010175). The vehicle moves forward, so each particle is smeared into a
    near-vertical streak. A pixel is taken as a particle when it is part of such a streak (narrow
    sideways, continuing straight down), stands out from its surroundings several times more than
    the seabed texture around it does (leaves, rock and sand are textured) and is not greener than
    neutral (seagrass leaves stand out green). Two sizes: sharp particles (~2-7 px wide) and the
    blurred ones close to the lens (~10-20 px wide, stricter). Runs on the GPU."""
    import torch.nn.functional as tf
    dev = GPU if GPU is not None else "cpu"
    full = torch.from_numpy(frame).to(dev).permute(2, 0, 1)[None].float()  # 1, BGR, H, W
    # streaks are found and the fill is worked out on a smaller copy (PARTICLE_WORK of the width,
    # 4x fewer pixels at 0.5); only the replacement itself is done at full size
    x = tf.interpolate(full, scale_factor=PARTICLE_WORK, mode="area") if PARTICLE_WORK < 1 else full
    h, w = x.shape[2:]
    sc = w / 1080  # sizes are for a 1080 px wide (portrait) frame

    def odd(n):
        return max(3, int(round(n * sc)) // 2 * 2 + 1)

    def box(img, k):  # k x k average as a horizontal then a vertical pass (same result, much faster)
        img = tf.avg_pool2d(img, (1, k), 1, (0, k // 2), count_include_pad=False)
        return tf.avg_pool2d(img, (k, 1), 1, (k // 2, 0), count_include_pad=False)

    lum = x.mean(1, keepdim=True)
    k9, k31 = odd(9), odd(31)
    texture = box((lum - box(lum, k9)).abs(), k31)

    def streaks(width, length, ratio, bright):
        kw, kl = odd(width), odd(length)

        def hopen(img):  # opening with a horizontal line: removes bright things narrower than kw
            low = -tf.max_pool2d(-img, (1, kw), 1, (0, kw // 2))
            return tf.max_pool2d(low, (1, kw), 1, (0, kw // 2))

        top = lum - hopen(lum)  # how much a narrow bright thing stands out
        excess = x - torch.cat([hopen(x[:, c:c + 1]) for c in range(3)], 1)
        red_green = excess[:, 2:3] - excess[:, 1:2]
        cand = (top > bright) & (top > ratio * texture) & (
            (red_green > -0.5) | ((red_green > 2) & (top > 2 * bright)))
        streak = -tf.max_pool2d(-cand.float(), (kl, 1), 1, (kl // 2, 0))  # continues straight down
        grow = (kl + 4) // 2 * 2 + 1, odd(width * 0.7)
        return tf.max_pool2d(streak, grow, 1, (grow[0] // 2, grow[1] // 2))[..., :h, :w] > 0

    mask = streaks(7, 9, PARTICLE_RATIO, PARTICLE_BRIGHT) | streaks(21, 31, PARTICLE_RATIO + 1, PARTICLE_BRIGHT)
    if not bool(mask.any()):
        return frame
    # fill from the nearest particle-free pixels: a small window, or a larger one where the small
    # window is mostly particles (a large streak), so no hole is left empty (black)
    keep = (~mask).float()
    big = odd(121)
    fill = box(x, big)  # last resort: a wide blur
    for size in (big, odd(41), odd(15)):  # smaller windows take over where they have enough free pixels
        cover = box(keep, size)
        mean = box(x * keep, size) / cover.clamp(min=1e-6)
        fill = torch.where(cover > (0.02 if size == big else 0.25), mean, fill)
    if PARTICLE_WORK < 1:
        size = full.shape[2:]
        mask = tf.interpolate(mask.float(), size=size, mode="nearest") > 0
        fill = tf.interpolate(fill, size=size, mode="bilinear", align_corners=False)
    out = torch.where(mask, fill, full)
    return out[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def deflicker_gain(frame, reset):
    """deflicker: per-channel factors (RGB) that bring this frame's average colour to its running
    average. In shallow water the waves at the surface focus sunlight into bands that sweep over
    the seabed. That light is a little redder than the diffuse light around it (GX010287 3:30: red
    +1.3 %, green +0.6 % per wave); invisible in the original, but the correction (green haze
    removed, red boosted ~2x) turned each wave into a red pulse. Slow changes (clouds, depth) stay."""
    if not DEFLICKER:
        return 1.0
    small = linear(cv2.resize(frame, (48, 85), interpolation=cv2.INTER_AREA))
    now = small.reshape(-1, 3).mean(0)
    if reset or flicker_state["level"] is None:
        flicker_state["level"] = now
    level = flicker_state["level"] = (1 - FLICKER_EMA) * flicker_state["level"] + FLICKER_EMA * now
    return np.clip(level / np.maximum(now, 1e-6), 1 / FLICKER_MAX, FLICKER_MAX).astype(np.float32)


def current_haze(frame, d, reset, altitude=np.nan, brightness=1.0):
    """natural: this frame's haze strength and how much to correct it (open water: less),
    both smoothed over time"""
    small = linear(cv2.resize(frame, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)) * brightness
    s = frame_haze(small, d)
    # less contrast than usual (open water, descent): full correction from CONTRAST_FULL of the
    # usual visibility up, none below CONTRAST_NONE
    amount = float(np.clip((visibility(small) / typical_visibility - CONTRAST_NONE)
                           / max(CONTRAST_FULL - CONTRAST_NONE, 1e-3), 0, 1))
    # high above the seabed the frame is mostly water: little seabed light is left to restore,
    # and the correction only turns the water magenta. Full up to ALT_FULL, none from ALT_NONE
    if ALT_NONE > ALT_FULL and np.isfinite(altitude):
        amount = min(amount, float(np.clip((ALT_NONE - altitude) / (ALT_NONE - ALT_FULL), 0, 1)))
    # a different camera white balance (see colour_look): full correction up to LOOK_KEEP,
    # none from LOOK_DROP on
    amount *= float(np.clip((LOOK_DROP - look_distance(small)) / (LOOK_DROP - LOOK_KEEP), 0, 1))
    last_s, last_amount = haze_state["scale"], haze_state["amount"]
    if not reset and last_s is not None:
        s = HAZE_EMA * s + (1 - HAZE_EMA) * last_s
        amount = HAZE_EMA * amount + (1 - HAZE_EMA) * last_amount
    haze_state["scale"], haze_state["amount"] = s, amount
    return s, amount


scene_state = {"wb": None}  # scene_balance: the colour balance in use (RGB, linear)


def scene_balance(out, first=False):
    """scene_balance: the pale surfaces of this frame (the brightest 2-8 %, as in the water fit)
    should be light beige; the balance that would make them so is followed slowly (SCENE_EMA),
    at most SCENE_MAX per channel, so warm or cool stretches are brought back without flicker."""
    small = linear(cv2.resize(out, (96, 170), interpolation=cv2.INTER_AREA)).reshape(-1, 3)
    lum = small.mean(1)
    lo_l, hi_l = np.quantile(lum, [0.92, 0.98])
    pale = small[(lum >= lo_l) & (lum <= hi_l)].mean(0)
    applied = scene_state["wb"]
    raw = pale / np.maximum(applied, 1e-6)  # as it would be without the balance in use
    want = NATURAL_SAND / np.maximum(raw / raw.mean(), 1e-6)
    want = np.clip(want / np.exp(np.log(want).mean()), 1 / SCENE_MAX, SCENE_MAX)
    k = 1.0 if first else SCENE_EMA
    scene_state["wb"] = np.exp((1 - k) * np.log(applied) + k * np.log(want)).astype(np.float32)


def enhance(frame, t, reset=False):
    if PARTICLE_METHOD == "blur":
        frame = remove_particles(frame)
    h, w = frame.shape[:2]
    k = deflicker_gain(frame, reset)  # 1 without deflicker
    if DEPTH_ANYTHING:
        d = pixel_range(frame, altitude_at(t), reset)
        haze, amount = current_haze(frame, d, reset, altitude_at(t), k)
        gain, b = correction(d, depth_at(t))
        # the haze and the light's colour change across the frame
        across = haze_columns(colour_across, d.shape[1])[None]
        gain, b = gain * across, b * haze_columns(haze, d.shape[1])[None] * across
        gain, offset = cv2.resize(gain, (w, h)), cv2.resize(-b - low / span, (w, h))
        gain, offset = on_device(gain), on_device(offset)
    else:
        rows = row_range(altitude_at(t), h)
        grid = np.repeat(rows[::max(1, h // 256), None], 144, axis=1)
        haze, amount = current_haze(frame, grid, reset, altitude_at(t), k)
        gain, b = correction(rows, depth_at(t))
        # small per-row and per-column factors; the full (h, w, 3) maps are built on the GPU
        gain, b, base = on_device(gain)[:, None, :], on_device(b)[:, None, :], on_device(low / span)
        across = on_device(haze_columns(colour_across, w))[None]
        gain = gain * across
        offset = -b * on_device(haze_columns(haze, w))[None] * across - base
    if DEFLICKER:  # = scaling the frame's channels by k; the haze removed stays the same
        gain = gain * on_device(k)
    if SCENE_BALANCE:
        if reset or scene_state["wb"] is None:  # first frame: measure once without, then apply
            scene_state["wb"] = np.ones(3, np.float32)
            scene_balance(apply_correction(frame, gain, offset, amount), first=True)
        wb = on_device(scene_state["wb"])
        gain, offset = gain * wb, offset * wb
    out = apply_correction(frame, gain, offset, amount)
    if SCENE_BALANCE:
        scene_balance(out)
    if STRENGTH < 1:
        out = cv2.addWeighted(out, STRENGTH, frame, 1 - STRENGTH, 0)
    return local_contrast(range_balance(out, reset))


# ==========================================
# 6. FIT REPORT + BEFORE / AFTER PREVIEW
# ==========================================

params = {
    "video": str(VIDEO), "method": METHOD, "range": RANGE, "tilt": TILT, "vfov": VFOV,
    "fit_on": FIT_ON, "backscatter_rgb": back_params, "attenuation_rgb": atten_params,
    "attenuation_fit_range_m": [fit_lo, fit_hi],
    "attenuation_points_used_rgb": [int(k) for k in kept.sum(0)], "attenuation_points": len(bin_d),
    "white_balance_rgb": white.tolist(),
    "depth_correction": DEPTH_CORRECTION, "depth_ref_m": depth_ref, "depth_slope_rgb": depth_slope.tolist(),
    "white_low": np.atleast_1d(low).tolist(), "white_span": np.atleast_1d(span).tolist(),
    "shadow_balance_rgb": shadow_k.tolist(),
}
info_path = RESULTS / f"{VIDEO.stem}_enhance_{NAME}.json"
info_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

colors = ["tab:red", "tab:green", "tab:blue"]
dd = np.linspace(0.5, MAX_RANGE, 200).astype(np.float32)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
show = np.random.default_rng(0).choice(len(dark_d), min(3000, len(dark_d)), replace=False)
for c in range(3):
    ax[0].scatter(dark_d[show], dark_rgb[show, c], s=2, alpha=0.2, color=colors[c])
    ax[0].plot(dd, backscatter(dd)[:, c], color=colors[c])
    ax[1].scatter(bin_d[kept[:, c]], bin_log[kept[:, c], c], color=colors[c])
    ax[1].scatter(bin_d[~kept[:, c]], bin_log[~kept[:, c], c], facecolors="none", edgecolors=colors[c])
    # the curve exactly as the correction uses it (held outside the fitted ranges)
    ax[1].plot(dd, predicted_log(dd)[:, c], color=colors[c])
    ax[2].plot(dd, gain_curve(dd)[:, c], color=colors[c])
ax[0].set(title="Backscatter (darkest pixels)", xlabel="range (m)", ylabel="linear intensity")
ax[1].set(title="Direct signal vs range (hollow = outlier, not used)", xlabel="range (m)",
          ylabel="log intensity")
ax[1].set_xlim(0, bin_d.max() * 1.1)
ax[2].set(title="Gain per channel (incl. sand white balance)", xlabel="range (m)", ylabel="x")
# legends: channel colours, and what the dots and lines in each panel are
from matplotlib.lines import Line2D
channels = [Line2D([], [], color=col, lw=2, label=name) for col, name in zip(colors, ["red", "green", "blue"])]
grey = "0.35"
ax[0].legend(handles=channels + [
    Line2D([], [], ls="none", marker="o", ms=4, color=grey, alpha=0.5, label="darkest 1 % of pixels"),
    Line2D([], [], color=grey, lw=2, label="haze removed")],
    fontsize=8, loc="lower right")
mid = [Line2D([], [], ls="none", marker="o", ms=6, color=grey, label="average signal at a range"),
       Line2D([], [], ls="none", marker="o", ms=6, mfc="none", color=grey, label="outlier, not used"),
       Line2D([], [], color=grey, lw=2, label="fitted curve")]
ax[1].legend(handles=channels + mid, fontsize=8, loc="best")
ax[2].legend(handles=channels, fontsize=8, loc="best")
fig.tight_layout()
fit_path = RESULTS / f"{VIDEO.stem}_enhance_{NAME}_fit.png"
fig.savefig(fit_path, dpi=100)
plt.close(fig)

tiles = []
for index in preview_picks[np.linspace(len(preview_picks) * 0.15, len(preview_picks) * 0.85, 4).astype(int)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
        continue
    small_size = fit_size(*frame.shape[:2], 720)
    frame = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA)
    clean = read_window(int(index), small_size) if DENOISE > 1 else frame
    rows = [frame, enhance(clean, index / fps, reset=True)]
    if DEPTH_ANYTHING:  # third row: the range map, yellow = near, dark blue = far
        d = cv2.resize(depth_state["range"], frame.shape[1::-1])
        shade = (255 * (1 - np.log(d) / np.log(MAX_RANGE))).clip(0, 255).astype(np.uint8)
        rows.append(cv2.applyColorMap(shade, cv2.COLORMAP_VIRIDIS))
    tiles.append(np.vstack(rows))
preview_path = RESULTS / f"{VIDEO.stem}_enhance_{NAME}_preview.jpg"
cv2.imwrite(str(preview_path), np.hstack(tiles))

print(f"Saved fit:     {fit_path}")
print(f"Saved values:  {info_path}")
print(f"Saved preview: {preview_path} (top = original, then enhanced"
      + (", bottom = range map: yellow near, dark far)" if DEPTH_ANYTHING else ")"))

if PREVIEW_ONLY:
    print("preview_only = yes: video not written. Set it to no to clean the whole video.")
    raise SystemExit


# ==========================================
# 7. CLEAN THE WHOLE VIDEO
# ==========================================

# Reading and writing run in background threads, so they overlap with the correction of the
# current frame. With ffmpeg (from the imageio-ffmpeg package or on the PATH) the 4K video is
# decoded and shrunk multi-threaded / on the NVIDIA decoder, and written with the NVIDIA encoder
# (h264_nvenc; libx264 without an NVIDIA GPU). Without ffmpeg, OpenCV reads and writes as before.

index = first if OUTSIDE == "cut" else 0
stop = last if OUTSIDE == "cut" else total
cap.set(cv2.CAP_PROP_POS_FRAMES, index)
ok, frame = cap.read()  # the size of the frames as OpenCV sees them (rotation applied)
if not ok:
    raise RuntimeError(f"Cannot read frame {index} of {VIDEO}")
cap.set(cv2.CAP_PROP_POS_FRAMES, index)
out_w, out_h = fit_size(*frame.shape[:2], OUTPUT_MAX)

try:
    import imageio_ffmpeg
    FFMPEG = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

ENCODER = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"] if GPU is not None
           else ["-c:v", "libx264", "-preset", "fast", "-crf", "18"])
print("Reading / writing: " + (f"ffmpeg ({'NVIDIA decoder + h264_nvenc' if GPU is not None else 'libx264'})"
                                if FFMPEG else "OpenCV (pip install imageio-ffmpeg for faster reading / writing)")
      + f", correction on the {'GPU' if GPU is not None else 'CPU'}")


def reader(frames):
    """put the frames index .. stop-1, shrunk to (out_w, out_h), into the queue; None at the end"""
    try:
        if FFMPEG:
            # OpenCV converts YUV to RGB with the BT.601 matrix, whatever the video says:
            # the same here, so the frames match those the water was fitted on
            cmd = [FFMPEG, "-v", "error", *(["-hwaccel", "cuda"] if GPU is not None else []),
                   *(["-ss", f"{(index - 0.5) / fps:.6f}"] if index > 0 else []),
                   "-i", str(VIDEO), "-an", "-sn", "-dn",
                   "-vf", f"scale={out_w}:{out_h}:flags=area:in_color_matrix=bt601,format=bgr24",
                   "-frames:v", str(stop - index), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            size = out_w * out_h * 3
            while True:
                buf = bytearray(size)  # writable, so the frame can go to PyTorch without a warning
                if proc.stdout.readinto(buf) != size:
                    break
                frames.put(np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3))
            proc.stdout.close()
            if proc.wait() != 0:
                print(f"ffmpeg stopped reading: {proc.stderr.read().decode(errors='replace')[-500:]}")
        else:
            for _ in range(stop - index):
                ok, f = cap.read()
                if not ok:
                    break
                if (out_w, out_h) != (f.shape[1], f.shape[0]):
                    f = cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_AREA)
                frames.put(f)
    finally:
        frames.put(None)


def writer_thread(frames):
    """write the frames from the queue until None"""
    if FFMPEG:
        # back to YUV with the BT.601 matrix (tagged as such): OpenCV reads it back unchanged
        cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
               "-r", f"{fps}", "-i", "-", "-vf", "scale=out_color_matrix=bt601:out_range=limited,format=yuv420p",
               *ENCODER, "-colorspace", "smpte170m", "-color_range", "tv", str(OUTPUT)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        while (f := frames.get()) is not None:
            proc.stdin.write(np.ascontiguousarray(f).data)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg could not write {OUTPUT}: {proc.stderr.read().decode(errors='replace')[-500:]}")
    else:
        out = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter.fourcc(*"mp4v"), fps, (out_w, out_h))
        while (f := frames.get()) is not None:
            out.write(f)
        out.release()


write_errors = []


def run_writer(frames):
    try:
        writer_thread(frames)
    except Exception as e:
        write_errors.append(e)
        while frames.get() is not None:  # keep taking frames, so the main loop doesn't hang
            pass


read_queue, write_queue = queue.Queue(maxsize=16), queue.Queue(maxsize=16)
threads = [threading.Thread(target=reader, args=(read_queue,), daemon=True),
           threading.Thread(target=run_writer, args=(write_queue,), daemon=True)]
for thread in threads:
    thread.start()

start = time.time()
done = 0
half = DENOISE // 2
window = deque()  # (frame number, frame): the frame being written and its neighbours
next_read, ended = index, False


def write(number, frames, centre):
    global done
    frame = frames[centre]
    if first <= number < last:
        write_queue.put(enhance(denoise(frames, centre), number / fps))
    else:
        write_queue.put(frame)
    done += 1
    if done % 300 == 0:
        speed = done / (time.time() - start)
        print(f"{done}/{stop - (first if OUTSIDE == 'cut' else 0)} frames, {speed:.1f} fps, "
              f"~{(stop - number) / speed / 60:.0f} min left")


while index < stop:
    # read ahead until the window holds `half` frames after the one to write
    while not ended and next_read <= min(index + half, stop - 1):
        frame = read_queue.get()  # already shrunk to the output size
        if frame is None:
            ended = True
            break
        window.append((next_read, frame))
        next_read += 1
    while window and window[0][0] < index - half:
        window.popleft()
    numbers = [n for n, _ in window]
    if index not in numbers:
        break
    write(index, [f for _, f in window], numbers.index(index))
    index += 1

write_queue.put(None)
threads[1].join()
cap.release()
if write_errors:
    raise write_errors[0]
if SRT.exists() and OUTSIDE == "cut" and first > 0:
    # the cut video starts at `first`: shift the GPS log so its times match the new video
    write_srt(srt_entries, OUTPUT.with_suffix(".srt"), offset=first / fps, until=(last - first) / fps)
elif SRT.exists():
    # so step 5 finds the GPS for the new video (with corrected times if the log needed them)
    write_srt(srt_entries, OUTPUT.with_suffix(".srt"))
print(f"Saved enhanced video: {OUTPUT} ({done} frames in {(time.time() - start) / 60:.1f} min)")
print(f"To use it, set video = {OUTPUT.as_posix()} in [extract] and/or [detect] of config.ini")
