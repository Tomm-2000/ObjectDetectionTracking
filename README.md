# Multi-Object Detection & Tracking with YOLO11

A video-based multi-object detection and tracking project using **YOLO11 Nano**, **Deep SORT**, and **OpenCV**. It detects objects in each video frame and overlays class labels, bounding boxes, and persistent track IDs.

## Pipeline

`Input video → YOLO11n detections → Deep SORT association → bounding boxes + class + track ID → MP4 output`

## Features

- Pretrained 80-class YOLO11n detector (no model training required)
- Deep SORT tracks detections across consecutive frames
- Per-frame object count and observed processing FPS on output video
- GPU automatically used when CUDA is available; CPU fallback
- Works with video files; no live camera required
- Configurable detection threshold and image size

## Google Colab (Tesla T4 recommended)

Clone this repository in a Colab session. Select a GPU runtime if you plan to pass `--device cuda`.

```python
!git clone https://github.com/Tomm-2000/ObjectDetectionTracking.git
%cd /content/ObjectDetectionTracking
%pip -q install -r requirements.txt
```

Fetch the public Intel sample video (not bundled in this repository):

```python
!wget -q https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4 -O /content/traffic_test.mp4
```

Run the independent script:

```python
!python -m src.track_video --video /content/traffic_test.mp4 --output /content/tracking_output.mp4 --device cuda
```

Preview or download the file:

```python
from IPython.display import Video, display
display(Video('/content/tracking_output.mp4', embed=True, width=640))
```

## Local usage

Install Python 3.12+ and the dependencies:

```bash
python -m pip install -r requirements.txt
python -m src.track_video --video path/to/video.mp4 --output outputs/tracking_output.mp4
```

If `ffmpeg` is installed, the script creates browser-friendly H.264; otherwise it retains an MP4V-encoded output. In Colab, `ffmpeg` is usually available.

## Inspect the output

The MP4 overlays class labels, track IDs, the visible track count, and observed processing FPS. Compare IDs over consecutive frames to see where the tracker maintains or loses an association. Use `python -m src.track_video --help` to inspect the available inference settings.

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests cover video dimension scaling and YOLO-to-Deep-SORT bounding-box conversion. They do **not** measure real-world tracking accuracy. The [original notebook](notebooks/CodeAlpha_ObjectDetectionTracking.ipynb) preserves the Colab experiment.

## Observed experiment

A Colab Tesla T4 video-processing run on Intel's public multi-object sample processed **647 frames** at an observed **51.64 processing FPS**. Visual inspection found stable IDs in the sampled footage; this is a qualitative observation, **not an IDF1/MOTA benchmark**. Processing FPS can change depending on hardware, video resolution and system load.

## Known limitations

- Object IDs may change following occlusion or loss of track.
- The default pretrained model only recognizes its supported object classes.
- Model weights and Deep SORT embedder weights may be downloaded on first run.
- `outputs/` videos and weights are excluded from Git to keep the repository lightweight.
- The provided pipeline processes video files; it is not a calibrated real-time webcam benchmark.

## Project structure

```text
ObjectDetectionTracking/
├── src/track_video.py
├── tests/test_track_video.py
├── notebooks/                # Original Colab experiment
├── outputs/                  # Generated video (ignored by Git)
├── requirements.txt
└── README.md
```

Source sample: [Intel IoT DevKit sample videos](https://github.com/intel-iot-devkit/sample-videos).

CodeAlpha Artificial Intelligence Internship Project.
