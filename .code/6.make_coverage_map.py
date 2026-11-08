import configparser
import sqlite3
import time
from pathlib import Path
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import LineString, shape
from lib.srt_log import describe, read_srt
from ultralytics import RTDETR, YOLO

# The project folder: this script's folder, or its parent when the script sits in the
# hidden .code/ folder (Binder version)
ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == ".code" else ROOT
config = configparser.ConfigParser()
config.read(ROOT / "config.ini", encoding="utf-8")
cfg, detect, enhance = config["coverage"], config["detect"], config["enhance"]


def setting(name, fallback):
    """[coverage] value, or the same setting from another section when left empty"""
    value = cfg.get(name, "").strip()
    return value if value else fallback.get(name).strip()


VIDEO = Path(setting("video", detect))
SRT = VIDEO.with_suffix(".srt")  # e.g. input/GX010287.srt
CONF = float(setting("conf", detect))
TILT = float(setting("tilt", enhance))  # degrees below horizontal (centre of the image)
VFOV = float(setting("vfov", enhance))  # degrees, top to bottom of the image
# degrees, left to right; empty = from VFOV and the frame's shape (lens without distortion)
HFOV = cfg.get("hfov", "").strip() or enhance.get("hfov", "").strip()
HFOV = float(HFOV) if HFOV else None
CLASSES = [c.strip() for c in (cfg.get("classes") or cfg.get("class") or "posidonia_oceanica").split(",") if c.strip()]
EVERY = cfg.getfloat("every")  # seconds between analysed frames
HEADING = cfg.get("heading", "track").strip().lower()  # track or a fixed compass direction
HEADING_OFFSET = cfg.getfloat("heading_offset")  # camera direction relative to the heading
LAYBACK = cfg.getfloat("layback")  # metres the camera is behind the GPS antenna
MAX_RANGE = cfg.getfloat("max_range")  # pixels farther than this are not mapped
CELL = cfg.getfloat("cell")  # grid cell size in metres
MIN_VIEWS = cfg.getint("min_views")  # cells seen in fewer frames stay empty
THRESHOLD = cfg.getfloat("threshold")  # % of the views that must be seagrass for a polygon
MIN_AREA = cfg.getfloat("min_area")  # smaller polygons are dropped (m2)
# ... per class, e.g. "debris: 0.1" (a tyre is ~0.3 m2, so min_area = 1 would drop every one)
CLASS_MIN_AREA = {k.strip(): float(v) for k, v in
                  (p.split(":") for p in cfg.get("class_min_area", "").split(",") if ":" in p)}

GRID_SIDE = 480  # each frame's mask is sampled on a grid with this longest side
HALF = torch.cuda.is_available()  # half precision only on the GPU: on the CPU it made the model ~150x slower

if not SRT.exists():
    raise RuntimeError(f"No .srt file found: {SRT}")


def find_model(choice):
    """'latest', a version like 'v2', or a path to a .pt file."""
    models = Path("models")
    if choice.lower() == "latest":
        versions = [d for d in models.glob("v*") if d.name[1:].isdigit() and (d / "model.pt").exists()]
        if not versions:
            raise RuntimeError("No trained models in models/. Run 3.train.py first.")
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


MODEL = find_model(setting("model", detect))
VERSION = MODEL.parent.name if MODEL.name == "model.pt" else MODEL.stem
NAME = f"{VIDEO.stem}_{VERSION}"  # e.g. GX010287_v7
model = load_detector(MODEL)
for name in [c for c in CLASSES if c not in model.names.values()]:
    print(f"Skipping class {name}: not in model {MODEL} ({', '.join(model.names.values())})")
CLASSES = [c for c in CLASSES if c in model.names.values()]
if not CLASSES:
    raise RuntimeError(f"None of the classes in [coverage] are in model {MODEL}: {list(model.names.values())}")
CLASS_IDS = {k: v for k, v in model.names.items() if v in CLASSES}  # model class id -> name
print(f"Video:  {VIDEO}")
print(f"Model:  {MODEL}, classes {', '.join(CLASSES)}")


# ==========================================
# 1. CAMERA POSITION, HEADING AND HEIGHT FROM THE SRT FILE
# ==========================================

entries = read_srt(SRT)  # timed by the logged clock when the subtitle times are wrong
print(f"SRT:    {describe(entries, SRT)}")
gps = pd.DataFrame([{"t": (e["start"] + e["end"]) / 2, "lat": e["lat"], "lon": e["lon"], "alt": e["alt"]}
                    for e in entries]).dropna(subset=["lat", "lon"]).reset_index(drop=True)

# metres in the local UTM zone
zone = int((gps.lon.mean() + 180) // 6) + 1
EPSG = (32600 if gps.lat.mean() >= 0 else 32700) + zone
to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
gps["e"], gps["n"] = to_utm.transform(gps.lon.to_numpy(), gps.lat.to_numpy())

# GPS positions jitter by a metre or so: smooth them over ~1.5 s for the position, ~3 s for
# the direction of travel. The altimeter jumps now and then: running median over ~2.5 s.
# Windows are in seconds, turned into entries (logs differ: 2 or 6 entries per second).
per_second = (len(gps) - 1) / max(gps.t.iloc[-1] - gps.t.iloc[0], 1e-6)


def window(seconds):
    return max(1, int(round(seconds * per_second)) // 2 * 2 + 1)  # odd, so it stays centred


gps["e_s"] = gps.e.rolling(window(1.5), center=True, min_periods=1).mean()
gps["n_s"] = gps.n.rolling(window(1.5), center=True, min_periods=1).mean()
gps["alt"] = gps.alt.rolling(window(2.5), center=True, min_periods=1).median()
e_dir = gps.e.rolling(window(3.5), center=True, min_periods=1).mean().to_numpy()
n_dir = gps.n.rolling(window(3.5), center=True, min_periods=1).mean().to_numpy()
ve, vn = np.gradient(e_dir, gps.t), np.gradient(n_dir, gps.t)
speed = np.hypot(ve, vn)
course = pd.Series(np.degrees(np.arctan2(ve, vn)))  # compass: 0 = north, 90 = east
course[speed < 0.15] = np.nan  # hardly moving: the direction is noise, keep the last one
course = np.unwrap(np.radians(course.ffill().bfill().to_numpy()))
print(f"GPS:    {len(gps)} positions, UTM zone {zone} (EPSG:{EPSG}), "
      f"median speed {np.median(speed):.2f} m/s, altitude {gps.alt.min():.1f}-{gps.alt.max():.1f} m")


def camera_at(t):
    """camera position (east, north), heading (radians from north) and height above the seabed"""
    heading = np.interp(t, gps.t, course) if HEADING == "track" else np.radians(float(HEADING))
    heading += np.radians(HEADING_OFFSET)
    e = np.interp(t, gps.t, gps.e_s) - LAYBACK * np.sin(heading)
    n = np.interp(t, gps.t, gps.n_s) - LAYBACK * np.cos(heading)
    return e, n, heading, float(np.interp(t, gps.t, gps.alt))


# ==========================================
# 2. WHERE EVERY PIXEL LANDS ON THE SEABED
# ==========================================
# Pinhole camera tilted TILT degrees below horizontal, flat seabed ALT metres below it.
# For a pixel with normalised image coordinates (x right, y down) the ray goes
#     down:    y * cos(tilt) + sin(tilt)
#     forward: cos(tilt) - y * sin(tilt)
#     right:   x
# and meets the seabed after ALT / down. Computed once for a height of 1 m, then scaled.

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
scale = GRID_SIDE / max(H, W)
gh, gw = round(H * scale), round(W * scale)

t_v = np.tan(np.radians(VFOV) / 2)
t_h = np.tan(np.radians(HFOV) / 2) if HFOV else t_v * W / H  # else: same focal length across and down
ys = ((np.arange(gh) + 0.5) / gh * 2 - 1) * t_v
xs = ((np.arange(gw) + 0.5) / gw * 2 - 1) * t_h
xn, yn = np.meshgrid(xs, ys)
tilt = np.radians(TILT)
down = yn * np.cos(tilt) + np.sin(tilt)
ahead = down > 0.02  # rays that reach the seabed (not the water above the horizon)
dist = np.where(ahead, 1 / np.maximum(down, 1e-6), 0)
FORWARD1 = dist * (np.cos(tilt) - yn * np.sin(tilt))
RIGHT1 = dist * xn
RANGE1 = dist * np.sqrt(xn ** 2 + yn ** 2 + 1)
print(f"Camera: tilt {TILT:g} deg, vertical FOV {VFOV:g} deg, horizontal FOV "
      f"{np.degrees(2 * np.arctan(t_h)):.0f} deg; mapping up to {MAX_RANGE:g} m from the camera")


# ==========================================
# 3. THE GRID
# ==========================================

times = np.arange(max(gps.t.min(), 0), min(gps.t.max(), total / fps), EVERY)
cams = [camera_at(t) for t in times]
pad = MAX_RANGE + 2
e0 = min(c[0] for c in cams) - pad
n0 = max(c[1] for c in cams) + pad
cols = int(np.ceil((max(c[0] for c in cams) + pad - e0) / CELL))
nrows = int(np.ceil((n0 - (min(c[1] for c in cams) - pad)) / CELL))
views = np.zeros(nrows * cols, np.float32)  # frames that saw each cell
cover = {name: np.zeros(nrows * cols, np.float32) for name in CLASSES}  # summed share of each class
colour = np.zeros((nrows * cols, 3), np.float32)  # summed colour, for the check mosaic
print(f"Grid:   {cols} x {nrows} cells of {CELL:g} m ({cols * CELL:.0f} x {nrows * CELL:.0f} m)")


def class_masks(frame):
    """0/1 map of each class on the sample grid"""
    result = model.predict(frame, conf=CONF, imgsz=640, half=HALF, verbose=False)[0]
    masks = {name: np.zeros((gh, gw), np.uint8) for name in CLASSES}
    if result.masks is not None:
        polys = result.masks.xy
    else:  # box-only models: the box counts as covered
        polys = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) for x1, y1, x2, y2 in result.boxes.xyxy.tolist()]
    fy, fx = gh / frame.shape[0], gw / frame.shape[1]
    for poly, cls in zip(polys, result.boxes.cls.tolist()):
        if int(cls) in CLASS_IDS and len(poly) >= 3:
            cv2.fillPoly(masks[CLASS_IDS[int(cls)]], [(np.asarray(poly) * [fx, fy]).astype(np.int32)], 1)
    return masks


def add_frame(frame, masks, cam):
    e, n, heading, alt = cam
    if not alt > 0:
        return
    use = ahead & (RANGE1 * alt <= MAX_RANGE)
    fwd, right = FORWARD1[use] * alt, RIGHT1[use] * alt
    pe = e + fwd * np.sin(heading) + right * np.cos(heading)
    pn = n + fwd * np.cos(heading) - right * np.sin(heading)
    col = ((pe - e0) / CELL).astype(np.int64)
    row = ((n0 - pn) / CELL).astype(np.int64)
    inside = (col >= 0) & (col < cols) & (row >= 0) & (row < nrows)
    cells, inverse = np.unique(row[inside] * cols + col[inside], return_inverse=True)
    count = np.bincount(inverse)
    # each frame counts once per cell: the share of its samples there that are each class
    views[cells] += 1
    for name, mask in masks.items():
        cover[name][cells] += np.bincount(inverse, weights=mask[use][inside]) / count
    small = cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA)[use][inside]
    for c in range(3):
        colour[cells, c] += np.bincount(inverse, weights=small[:, c]) / count


# ==========================================
# 4. RUN THROUGH THE VIDEO
# ==========================================

start = time.time()
targets = set(int(round(t * fps)) for t in times)
frame_times = {int(round(t * fps)): t for t in times}
index, done = 0, 0
while index <= max(targets):
    if index not in targets:
        if not cap.grab():
            break
        index += 1
        continue
    ok, frame = cap.read()
    if not ok:
        break
    if max(frame.shape[:2]) > 1920:  # the model works at 640 px; 4K only slows it down
        frame = cv2.resize(frame, (W * 1920 // max(H, W), H * 1920 // max(H, W)), interpolation=cv2.INTER_AREA)
    add_frame(frame, class_masks(frame), camera_at(frame_times[index]))
    index += 1
    done += 1
    if done % 100 == 0:
        speed_fps = done / (time.time() - start)
        print(f"{done}/{len(targets)} frames, ~{(len(targets) - done) / speed_fps / 60:.0f} min left")
cap.release()
print(f"Analysed {done} frames in {(time.time() - start) / 60:.1f} min")


# ==========================================
# 5. SAVE: COVERAGE RASTER, CHECK MOSAIC, POLYGONS
# ==========================================

# output_dir: where the maps go (default results/coverage; the notebook gives each run its own folder)
RESULTS = Path(cfg.get("output_dir", "").strip() or Path("results") / "coverage")
RESULTS.mkdir(parents=True, exist_ok=True)
transform = from_origin(e0, n0, CELL, CELL)
crs = f"EPSG:{EPSG}"
seen = views >= MIN_VIEWS
coverage = {name: np.where(seen, 100 * cover[name] / np.maximum(views, 1), -1).reshape(nrows, cols).astype(np.float32)
            for name in CLASSES}

for name in CLASSES:
    with rasterio.open(RESULTS / f"{NAME}_{name}_coverage.tif", "w", driver="GTiff", height=nrows, width=cols,
                       count=1, dtype="float32", crs=crs, transform=transform, nodata=-1, compress="deflate") as dst:
        dst.write(coverage[name], 1)
        dst.set_band_description(1, f"{name} cover % (share of the frames that saw the cell)")

mosaic = np.where(seen[:, None], colour / np.maximum(views, 1)[:, None], 0).astype(np.uint8)
mosaic = mosaic.reshape(nrows, cols, 3)[..., ::-1].transpose(2, 0, 1)  # BGR -> RGB bands
mosaic_path = RESULTS / f"{NAME}_mosaic.tif"
with rasterio.open(mosaic_path, "w", driver="GTiff", height=nrows, width=cols, count=3, dtype="uint8",
                   crs=crs, transform=transform, nodata=0, compress="deflate", photometric="RGB") as dst:
    dst.write(mosaic)


def polygons(binary):
    geoms = [shape(g) for g, v in shapes(binary.astype(np.uint8), mask=binary, transform=transform) if v == 1]
    # cell staircase edges -> slightly smoother outlines
    return [g.simplify(CELL * 0.75) for g in geoms]


def class_patches(name):
    patches = gpd.GeoDataFrame(geometry=polygons(coverage[name] >= THRESHOLD), crs=crs)
    patches["area_m2"] = patches.area.round(2)
    patches = patches[patches.area_m2 >= CLASS_MIN_AREA.get(name, MIN_AREA)].reset_index(drop=True)
    patches.insert(0, "id", range(1, len(patches) + 1))
    patches.insert(1, "class", name)
    return patches


patches = {name: class_patches(name) for name in CLASSES}
surveyed = gpd.GeoDataFrame(geometry=polygons(seen.reshape(nrows, cols)), crs=crs)
surveyed["area_m2"] = surveyed.area.round(1)
track = gpd.GeoDataFrame({"video": [VIDEO.name]},
                         geometry=[LineString([(c[0], c[1]) for c in cams])], crs=crs)

gpkg_path = RESULTS / f"{NAME}_coverage.gpkg"
# start a fresh file: mode="w" only replaces the first layer, the "a" layers below would
# otherwise be added to those of earlier runs (close it in QGIS before rerunning)
gpkg_path.unlink(missing_ok=True)
surveyed.to_file(gpkg_path, layer="surveyed_area", driver="GPKG", mode="w")
track.to_file(gpkg_path, layer="camera_track", driver="GPKG", mode="a")
for name in CLASSES:
    patches[name].to_file(gpkg_path, layer=name, driver="GPKG", mode="a")
    patches[name].to_file(RESULTS / f"{NAME}_{name}.shp")

# QGIS styles: styles/<layer>.qml (made in QGIS: Layer Properties > Style > Save Style, or
# saveNamedStyle). Each one becomes the layer's default style in the GeoPackage (its
# layer_styles table, laid out as QGIS saves it) and a .qml next to the shapefile, so QGIS
# opens the layers already styled.
STYLES = ROOT / "styles"
styled = [name for name in ["surveyed_area", "camera_track", *CLASSES] if (STYLES / f"{name}.qml").exists()]
if styled:
    with sqlite3.connect(gpkg_path) as db:
        db.execute("""CREATE TABLE layer_styles ("id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            "f_table_catalog" TEXT(256), "f_table_schema" TEXT(256), "f_table_name" TEXT(256),
            "f_geometry_column" TEXT(256), "styleName" TEXT(30), "styleQML" TEXT, "styleSLD" TEXT,
            "useAsDefault" BOOLEAN, "description" TEXT, "owner" TEXT(30), "ui" TEXT(30),
            "update_time" DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))""")
        db.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, description, last_change) "
                   "VALUES ('layer_styles', 'attributes', 'layer_styles', '', strftime('%Y-%m-%dT%H:%M:%fZ','now'))")
        for name in styled:
            qml = (STYLES / f"{name}.qml").read_text(encoding="utf-8")
            column = db.execute("SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?", (name,)).fetchone()
            db.execute("INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, f_geometry_column, "
                       "styleName, styleQML, styleSLD, useAsDefault, description, owner) VALUES ('', '', ?, ?, ?, ?, '', 1, ?, '')",
                       (name, column[0] if column else "", name, qml, f"styles/{name}.qml"))
            if name in CLASSES:
                (RESULTS / f"{NAME}_{name}.qml").write_text(qml, encoding="utf-8")
    db.close()

surveyed_m2 = surveyed.area.sum()
print(f"Saved coverage rasters: {RESULTS / NAME}_<class>_coverage.tif  (0-100 %, {CELL:g} m cells, EPSG:{EPSG})")
print(f"Saved check mosaic:     {mosaic_path}  (frames placed on the map, to check the alignment)")
print(f"Saved polygons:         {RESULTS / NAME}_<class>.shp")
print(f"                        {gpkg_path}  (layers {', '.join(CLASSES)}, surveyed_area, camera_track)")
if styled:
    print(f"QGIS styles:            {', '.join(styled)} (from {STYLES.name}/, in the GeoPackage and as .qml next to the shapefiles)")
print(f"Surveyed {surveyed_m2:,.0f} m2")
for name in CLASSES:
    area = patches[name].area.sum()
    shown = f"{area:,.0f}" if area >= 10 else f"{area:.2f}"  # small areas (debris) with decimals
    print(f"  {name} (>= {THRESHOLD:g} % of views): {shown} m2 = {100 * area / max(surveyed_m2, 1):.1f} %, "
          f"{len(patches[name])} patches")
