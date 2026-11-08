# Posidonia oceanica mapping from AUV video

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nikolaoskokkos/Efxinnos_ML_videos/HEAD?labpath=posidonia_pipeline.ipynb)

A Jupyter notebook that maps the seagrass *Posidonia oceanica* from a GoPro video and its `.srt`
log (GPS position, depth and height above the seabed).

1. **Enhance** the video: the water haze and green cast are removed, the sand is balanced to beige
   and the far part of the frame is given the same colour as the near seabed.
2. **Detect** *Posidonia oceanica* and rock with a trained YOLO11s-seg model (the **latest** by
   default; v8 to v13 to choose from; v12 and v13 also find debris such as tyres and plastic). A live view
   shows the latest annotated frame while it runs, and a frame browser steps through the results
   afterwards.
3. **Map** the coverage: every analysed frame is placed on the seabed, giving coverage rasters
   (GeoTIFF) and polygons (GeoPackage / shapefile) in UTM, plus an interactive map.

Two 30 s samples (1080 x 1920) with their `.srt` logs are included, so the notebook works straight
away:

| Sample | Part of the dive | What it shows |
| --- | --- | --- |
| `GX010287_sample.mp4` | GX010287, 1:30-2:00 | Seagrass and rock, 4 to 5.5 m deep, 4.3 to 5.1 m above the seabed |
| `GX010175_sample.mp4` | GX010175, 1:20-1:50 | A seagrass meadow with sand patches in very shallow, green water (about 1 m deep, 4 m above the seabed) |

The notebook starts on `GX010287_sample.mp4`; pick the other one in step 0.

## Quick start

1. Press the **launch binder** badge above and wait for the notebook to open. The first start after
   a change to the repository takes 5 to 10 minutes; later ones take under a minute.
2. Run the cells from top to bottom with **Shift + Enter**. Each step shows a settings panel; set
   the values, then run the cell below the panel.
3. The defaults are the recommended settings: for a first try, just run every cell.
4. In the last cell, choose what to download and press **Make the zip**, then download
   `results.zip` before you close the tab. Binder deletes the
   session when you leave.

Each 30 s sample takes about 8 to 10 minutes for the three steps on Binder.

## Using the notebook

### First cell: check the environment

Shows the Python, PyTorch and OpenCV versions, the number of CPU cores, whether there is a GPU
(not on Binder), whether ffmpeg was found, and the videos in `input/`.

### Step 0: choose the video

The **Video** list shows the videos in `input/`. It starts on the video you used last (at first
`GX010287_sample.mp4`). The enhanced videos the notebook makes are not listed. Under the list you see the size, length, frame rate and whether the `.srt` log was found.
The log must have the same name as the video, for example `dive.mp4` and `dive.srt`.

To add your own video and its `.srt`, use one of these:

- **Upload** button: for small files (.mp4, .mov, .srt). You can select several files at once.
- **File browser** on the left: for big files, open `input/` and drag the files into it.
- **Download URL**: a direct link to a `.mp4` or `.srt`, then **Download**.

Press **Refresh** to see the new files in the list.

Without a `.srt`, step 1 uses a fixed altitude (5 m) and step 3 can't be run.

### Step 1: enhance the video

Run the panel cell, set the values, then run the cell below it. When it's done, the notebook shows
a before / after picture at four moments (original on top, enhanced below) and where the enhanced
video was saved. With Depth Anything, the picture has a third row showing the distance map
(yellow = near, dark blue = far).

| Control | Default | What it does |
| --- | --- | --- |
| Enhance the video | ticked | Untick it to skip the enhancement and detect on the original video. |
| Start / End | empty | Only enhance part of the video, in seconds (`90`) or minutes:seconds (`1:30`). Empty = from the beginning / to the end. The rest is left out, the `.srt` is cut to match and the water is fitted on that part only. |
| Distance to the seabed | flat seabed | How far the seabed is from the camera, which sets how much colour and haze is corrected. **Flat seabed** works it out from the `.srt` altitude and the camera tilt, as if the seabed were flat (fast). **Depth Anything V2** estimates the distance of every pixel from the image, which helps with slopes, big rocks and open water. It is several times slower on a CPU and downloads its model (about 100 MB) the first time. Its estimate is tied to the flat-seabed distances (up to ±30 % closer or farther), because on its own it puts the top of the frame 30 to 40 m away. |
| Strength | 1.0 | 1 = full correction, 0.5 = halfway between the original and the corrected video. 0.75 suits very green, shallow videos. |
| Haze removal | 1.0 | 1 = as fitted from the video. Above 1 gives a clearer picture, but the darkest seagrass can turn grey-purple. Below 1 is softer. |
| Far colour limit | 1.0 (off) | Limits how much the colour may change with distance. Try 1.3 if the top of the frame still turns yellow or orange. |
| Distance colour balance | 1.0 | The far (top) part of the frame comes out yellower, browner or greener and flatter than the near part, because its light travelled further through the water. This gives the far sand and rock the colour, brightness and fine detail of the near seabed. Smooth open water is left as it is. 0 = off. |
| Local contrast | 0 (off) | Brings out the texture of seagrass, rock and sand. 2 = clearly sharper. It changes only the brightness, not the colour, and makes the picture a little darker. |
| Sand brightness (L) | 80 | How bright the sand and pale rock are made (0 to 100). Lower it (75) if the video looks whitish; raise it if it looks dark. |
| Shadow balance | ticked | Keeps the dark seagrass olive instead of pink or mauve. |
| Correct for camera depth | ticked | Uses the depth below the surface (`imm.` in the `.srt`): deeper water has less red light. |
| Remove floating particles | unticked | Removes the floating particles, which the correction turns into orange streaks, with the method below. Enhance the videos you detect on the same way as the frames the model was trained on. |
| Removal method | median | Shown when *Remove floating particles* is ticked. **median** lines up the 2 frames before and the 2 after each frame and takes the middle value of every pixel. A drifting particle is missing from most of them, so it disappears, but the seabed gets slightly softer and fish fade. It removed about 9 in 10 moving particles on GX010175, is about 3 times slower, and adds `_pmedian5` to the file name. **bright** (`_pbright5`) replaces only what is brighter than in those frames: the seabed stays sharper but more particles are left. **blur** (`_pblur`) fills streaks with the colour around them: fast, but it can leave smudges. |
| Saved video size | 1920 px | Longest side of the enhanced video. Smaller files are a bit faster to process in steps 2 and 3. |
| Frames to fit the water | 60 | Frames sampled across the video to fit the water. More frames give a steadier fit and take longer to start. |
| Preview only | unticked | Only fits the water and saves the before / after picture, without making the video. It is fast, so use it to try settings. |

The enhanced video is saved in the run's folder as
`results/<run>/enhancement/<video>_enhanced_<method>.mp4`, with a copy of its `.srt` (see
[Output files](#output-files)).
The method part of the name is `natural` or `natural_depth` (Depth Anything). Particle removal
adds `_pmedian5`, `_pbright5` or `_pblur`, and local contrast adds `_lc<value>`, for example
`GX010175_sample_enhanced_natural_pmedian5.mp4` or `GX010287_sample_enhanced_natural_depth_lc2.mp4`.
Running step 1 again starts a new run folder, so earlier videos are kept.

#### Getting the most natural picture

The defaults (flat seabed, distance colour balance 1.0, everything else as shown above) gave the
most natural and steady result on the test dives. They were also 4 times faster than Depth
Anything. If something still looks wrong, change one setting at a time with **Preview only**
ticked:

| What you see | What to change |
| --- | --- |
| Too white or milky | Sand brightness 75; haze removal no higher than 1.0 |
| Still green or blue | Strength 1.0; haze removal 1.1 |
| Top of the frame yellow, brown or orange | Keep distance colour balance at 1.0 and set far colour limit to 1.3 |
| Top of the frame slightly pink or mauve | Distance colour balance 0.7 |
| Dark seagrass grey-purple or pink | Haze removal 1.0 or lower; shadow balance ticked |
| Texture looks flat | Local contrast 1 to 2 |
| Steep slopes or big rocks look wrong far away | Distance to the seabed: Depth Anything V2 |
| Floating particles or orange streaks in the water | Tick *Remove floating particles*, method *median* |

### Step 2: detect the seagrass

The model finds *posidonia_oceanica* and *rock* (and, with v12 or v13, *debris*) in the video made
by step 1, or in the original video if step 1 was skipped.

| Model | Classes | Trained on | mask mAP50 | Use it for |
| --- | --- | --- | --- | --- |
| **v8** | posidonia, rock | GX010287 | 0.703 | GX010287-like videos |
| v10 | posidonia, rock | GX010287 | 0.602 | (older, scores lower) |
| **v11** | posidonia, rock | GX010287 + GX010175 | 0.682 | GX010175-like videos: on a meadow stretch it found seagrass in 93 % of the frames, against 30 % for v8 |
| v12 | posidonia, rock, debris | GX010287 + GX010175 | 0.610 | (superseded by v13: only 5 debris examples) |
| **v13** | posidonia, rock, debris | GX010287 + GX010175, + extra debris frames | 0.674 | Finding debris (tyres, plastic). It finds less seagrass than v11, because some of its training frames had only the debris labelled, so use v11 for the seagrass |

Each model's mAP is measured on its own validation frames, so the scores of models trained on
different frames can't be compared directly. v11 to v13 were trained on GX010175 frames enhanced
with *Remove floating particles* (method *median*), so enhance GX010175 videos the same way for them.

| Control | Default | What it does |
| --- | --- | --- |
| Model | latest | The models above, newest first, listed with their classes and mask mAP50. The newest is preselected and shown as *latest (vN)*, for example *latest (v13)*; the label and the preselection move on by themselves when a newer model is added to `models/`. |
| Confidence | 0.40 | Lower finds more faint seagrass, but gives more false detections. |
| Colour opacity | 0.30 | How strongly the detected areas are coloured in the saved video: 0 = only the outline, 1 = solid colour. |
| Video saturation | 1.00 | Colour boost of the saved video and the live view only: 1 = unchanged, 1.3 = livelier, 0 = black and white. The model always sees the unchanged frame, so this doesn't change the detections. |
| Every n-th frame | 2 | Run the model on every n-th frame. 2 = half the frames and twice as fast. The saved video then plays at the matching lower frame rate. |
| Saved video size | 1080 px | Longest side of the annotated video. |
| Live view updates | every 3rd frame | How often the live view is refreshed. Off is a little faster. |

While it runs, the cell shows the latest annotated frame, what was found in it and a small chart
of the cover so far. To stop the detection, press the stop button (■) in the toolbar.

When it's done, you see four frames of the annotated video, a chart of the detected cover (% of the
frame) over time, the average cover per class and a player for the annotated video.

**Browse the detections** (the next cell) steps through the annotated video frame by frame with
the slider, or press play. On the right are the areas found in that frame, the best confidence, the
% of the frame each class covers and the frame's place on the cover chart.

### Step 3: coverage map

Every analysed frame is placed on the seabed using the GPS track (the direction of travel is taken
as the camera direction), the altitude from the `.srt` and the camera geometry, assuming a flat
seabed. Each map cell gets the % of the frames that saw it in which the class was detected. This
step uses the model and confidence chosen in step 2.

| Control | Default | What it does |
| --- | --- | --- |
| Seconds between frames | 0.5 | Time between the frames that are mapped. Smaller = more frames, slower. |
| Cell size (m) | 0.25 | Size of the map cells. |
| Max range (m) | 8 | Pixels farther from the camera than this are not used, because they are small and less accurate. |
| Min. views per cell | 3 | Cells seen in fewer frames than this stay empty. |
| Class threshold (%) | 50 | A cell becomes part of a polygon of that class when at least this % of the frames that saw it found the class. |
| Min. polygon (m2) | 1 | Smaller polygons are dropped. |
| Camera tilt / Vertical FOV / Horizontal FOV | 64° / 75.3° / 46.2° | GoPro HERO13 4K Wide behind a flat port, portrait video. Change them for another camera or mount. They are also used by step 1 the next time it runs. |

When it's done, the notebook shows:

- a figure with the video frames placed on the map (to check the alignment) and a coverage raster
  per class, with the polygon outlines and the camera track in red;
- a table with the surveyed area and, per class, the number of patches, their area in m² and their
  share of the surveyed area;
- an interactive map with OpenStreetMap or satellite imagery underneath, and layers for the surveyed
  area, each class and the camera track (switch them on and off at the top right). They are drawn
  with the QGIS styles stored in the GeoPackage (seagrass blades, rock texture, debris hatching and
  triangles, dotted track), so a style changed in `styles/` shows here too. Hover over a polygon to
  see its class and area. The satellite imagery is detailed only up to zoom level 18 at the test
  site; closer than that it is just enlarged.

### Step 4: download the results

The last cell shows a download panel:

- **Runs:** which runs to include. It starts on the newest. Each run is listed with its date, time
  and video, and what was made in it, for example
  `2026-11-08 10:43 GX010175_sample: enhanced, detected (v13), mapped`.
- **What:** which kinds of results to include: enhanced videos, enhancement previews and charts,
  detection videos, detection tables (CSV), coverage maps (GeoTIFF, GeoPackage, shapefiles),
  settings used.

Both lists have an **all** option. Choose several with Ctrl + click or Shift + click. Then press
**Make the zip**: it makes `results.zip` and shows a download link. The videos are the big
files, so leave them out for a small zip.

## Output files

Each run has a folder of its own, `results/<date>_<time>_<video>/`, for example
`results/2026-11-08_104314_GX010175_sample/`. Step 1 starts a new run, and steps 2 and 3 add to
it. If you start with step 2 (after a kernel restart, for example), it starts a run itself.

| File in `results/<run>/` | Made by | Contents |
| --- | --- | --- |
| `settings.ini` | Steps 1 to 3 | The settings the run used (a copy of `config.ini`) |
| `enhancement/<video>_enhanced_<method>.mp4` (+ `.srt`) | Step 1 | The enhanced video and its log |
| `enhancement/<video>_enhance_<method>_preview.jpg` | Step 1 | Before (top) and after (bottom) at four moments |
| `enhancement/<video>_enhance_<method>_fit.png` | Step 1 | Chart of the water fit (haze and colour loss with distance) |
| `prediction/<video>_<model>_prediction.mp4` | Step 2 | The video with the detected areas coloured, their boxes and confidence |
| `prediction/<video>_<model>_detections.csv` | Step 2 | One row per analysed frame: `frame`, `time_s`, and per class `<class>_count` (areas), `<class>_max_conf` and `<class>_cover_pct` (% of the frame) |
| `coverage/<video>_<model>_<class>_coverage.tif` | Step 3 | Cover in % per cell (GeoTIFF, UTM) |
| `coverage/<video>_<model>_coverage.gpkg` | Step 3 | Layers: one polygon layer per class (with `class` and `area_m2`), `surveyed_area` and `camera_track`. Open it in QGIS: the layers come up already styled (see below). |
| `coverage/<video>_<model>_<class>.shp` (+ `.qml`) | Step 3 | The same polygons as shapefiles, each with its QGIS style next to it |
| `coverage/<video>_<model>_mosaic.tif` | Step 3 | The video frames placed on the map, to check the alignment |

`results.zip` (step 4) is made in the notebook's folder.

**QGIS styles.** All the GeoPackage layers (and the shapefiles) open in QGIS with the styles in
`styles/`:

- seagrass: a green fill with blades;
- rock: a rock texture with a grey outline;
- camera track: a black dotted line;
- surveyed area: a pale, semi-transparent sand fill with a dashed grey outline, so the mapped
  seabed with no seagrass or rock shows up;
- debris (for a model trained with a debris class): an orange fill with dark red hatching and
  outline, plus an orange triangle at the centre of each patch, so even a tyre-sized patch shows
  when zoomed out.

 The styles are stored in the GeoPackage as each layer's default style, and as
a `.qml` file next to each shapefile. To change a style, style the layer in QGIS, then use *Layer
Properties > Style > Save Style* to save it over `styles/<layer>.qml` (for example
`styles/rock.qml`), using the layer names `posidonia_oceanica`, `rock`, `debris`, `camera_track` or
`surveyed_area`. Embed any image the style uses, as the rock texture is, so the style works on
other computers. The next run of step 3 uses the new style.

`<video>` is the name of the video that was analysed, for example `GX010287_sample_enhanced_natural`,
and `<model>` is the model (`v8`, `v11`, `v13`, ...). Results made with different settings or
models are kept side by side, so you can compare them. The debris files appear only with a model
that knows debris (v12, v13). Debris patches are kept from 0.1 m² (`class_min_area` in
`config.ini`), because a tyre is only about 0.3 m², while seagrass and rock patches need 1 m².

## Tips

- **Try settings quickly.** Tick *Preview only* in step 1, or set a short start / end, then run
  the whole video once you like the picture.
- **After changing step 1, run steps 2 and 3 again**, because the detection runs on the enhanced
  video.
- **After a kernel restart** (or reopening the session), run the cells from the top again: step 2
  uses the video that step 1 made in the same session, and otherwise falls back to the original.
- **To see earlier results without running the steps again**, use these in a new cell (with the
  names of your files). They look in the newest run folder that has those results; add
  `run="2026-11-08_104314_GX010175_sample"` for a particular run:

  ```python
  browse_detections("GX010175_sample_enhanced_natural_pmedian5", "v13")  # step 2: the enhanced video's name, the model
  show_map("GX010175_sample_enhanced_natural_pmedian5_v13")              # step 3: <video>_<model>
  ```

- **More settings.** `config.ini` holds every setting the scripts use. It is made from
  `config_template.ini` at the first run and updated with the panel values each time a step runs.
  Settings without a control in the panels can be changed there and are kept, for example:
  - `depth_model`: the Depth Anything model (`...-Small-hf`, `...-Base-hf`, `...-Large-hf`). The larger
    ones are much slower and gave the same far distances in our tests.
  - `depth_fit`: how Depth Anything's estimate is tied to the flat-seabed distances. `relief` (the
    default) keeps its bumps and dips; `curve` only bends the flat-seabed distances to its overall
    trend.
  - `altitude`: the altitude in metres for videos without a `.srt` (default 5).
  - `outside`: what happens outside start / end. `cut` (the default) leaves it out; `keep` copies
    those frames unchanged.
  - `fit_on`: what the water is fitted on. `part` (the default) uses only start to end; `all` uses
    the whole video.
  - `particle_frames`: how many frames particle removal (*median* or *bright*) combines
    (default 5: 2 before and 2 after; makes the file name `_pmedian<n>` or `_pbright<n>`).
  - `particle_step`, `particle_grow`: for the *bright* method, how much brighter than the
    neighbouring frames counts as a particle (default 5, out of 255) and how far around each one
    is replaced too (default 9 px).
  - Older settings files with `denoise_frames` / `denoise_mode` still work: they are read as
    particle removal with the *median* (`all`) or *bright* (`particles`) method.

## Troubleshooting

| Problem | Solution |
| --- | --- |
| A change to the code or a new setting doesn't appear | Restart the kernel (*Kernel > Restart*) and run the cells from the top. Running the import cell again isn't enough, because Python keeps the module it already loaded. |
| "no .srt with the same name" | Give the log the video's name (`dive.mp4` and `dive.srt`) and put it in `input/`. |
| Step 3: "The coverage map needs ... .srt" | The coverage map needs the GPS and altitude from the `.srt`. Add it and run step 1 again, which copies it next to the enhanced video. |
| Depth Anything stops with a download error | Its model comes from Hugging Face the first time; try again, or use flat seabed. |
| The session disconnected | Binder stops after about 10 minutes without activity and its files are lost. Start it again from the badge and download `results.zip` before you leave. |
| Everything is slow | Use a shorter start / end, tick *Preview only* while trying settings, use flat seabed instead of Depth Anything, set *Every n-th frame* to 3 or more, or set *Live view updates* to off. |

## Limits on Binder

- **No GPU.** Everything runs on the CPU (1 to 2 cores, 2 to 8 GB RAM). Each 30 s sample takes about
  8 to 10 minutes for the three steps (enhancement about 4 minutes, detection about 3, map under
  1). Use short clips of your own videos (under 1 to 2 minutes, or set start / end in step 1), or
  run the full project on a computer with an NVIDIA GPU for whole dives.
- **Uploads.** Small files can be uploaded with the button in the notebook; drag bigger ones into
  `input/` in the file browser on the left. A 4K GoPro file is several GB, so cut and shrink it
  first, for example 60 s from 1:30 at 1080 x 1920:
  `ffmpeg -ss 90 -t 60 -i GX010287.MP4 -an -vf scale=1080:1920 -c:v libx264 -crf 24 clip.mp4`.
  Cut the `.srt` to the same part too (if you upload the whole video instead, the start / end
  settings of step 1 do this for you).
- **Sessions are temporary.** Binder deletes the session when you close it or after about 10
  minutes without activity. Download `results.zip` (last cell) before you leave.

## The `.srt` log

Each entry gives the clock time, position, depth below the surface (`imm.`, m) and height above
the seabed (`alt.`, m), for example:

```text
12
00:00:05,500 --> 00:00:06,000
2026/07/16-10:48:05 -- lat:40.828141 lon:24.340483 imm.:4.012220 alt.:12.429604 -- light state:1
```

Without a `.srt`, step 1 uses a fixed altitude (`altitude` in `config.ini`) and step 3 can't be
run.

## Files

| File | Purpose |
| --- | --- |
| `posidonia_pipeline.ipynb` | The notebook |
| `config_template.ini` | Default settings, with a comment for each; the notebook makes `config.ini` from it with the chosen values |
| `styles/` | QGIS styles (`.qml`) for the map layers, added to the results of step 3 |
| `models/v8/`, `v10/`, `v11/`, `v12/`, `v13/` | The trained models (`model.pt`, YOLO11s-seg: posidonia_oceanica, rock; v12 and v13 also debris) and their `info.txt` (training data and scores); chosen in step 2 |
| `input/` | Videos and their `.srt` logs (same name): the samples `GX010287_sample` and `GX010175_sample` (`.mp4` + `.srt`) |
| `requirements.txt` | Python packages (CPU-only PyTorch) |
| `apt.txt` | System packages: ffmpeg, and the libraries OpenCV needs |
| `runtime.txt` | Python version |

The notebook makes `config.ini`, `results/` (one folder per run) and `results.zip`; these are in
`.gitignore` (as is `input/<video>_enhanced_*`, where earlier versions saved the enhanced videos).
