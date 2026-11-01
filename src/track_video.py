"""Detect and track multiple objects in a video with YOLO11 + Deep SORT.

Usage:
    python -m src.track_video --video input.mp4 --output output.mp4

Install dependencies first using: pip install -r requirements.txt
"""

import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path


def scaled_size(width: int, height: int, max_width: int = 640) -> tuple[int, int]:
    """Keep aspect ratio, never upscale, and return even dimensions."""
    if width <= 0 or height <= 0 or max_width < 2:
        raise ValueError("Video dimensions and max-width must be positive.")
    scale = min(1.0, max_width / width)
    out_width = max(2, int(width * scale) // 2 * 2)
    out_height = max(2, int(height * scale) // 2 * 2)
    return out_width, out_height


def make_detection(x1, y1, x2, y2, confidence, class_name):
    """Convert a YOLO xyxy box to Deep SORT's (ltwh, confidence, class)."""
    box_width = float(x2) - float(x1)
    box_height = float(y2) - float(y1)
    if box_width <= 0 or box_height <= 0:
        return None
    return ([float(x1), float(y1), box_width, box_height],
            float(confidence), str(class_name))


def process_video(args):
    # Import heavyweight libraries only when processing video, not during unit tests.
    import cv2
    import torch
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort

    source = Path(args.video).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    if source == destination:
        raise ValueError("Input and output paths must be different.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available() and args.device != "cpu"
    if args.device == "cuda" and not use_gpu:
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")

    model = YOLO(args.model)
    tracker = DeepSort(max_age=args.max_age, n_init=3, embedder_gpu=use_gpu)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = scaled_size(width_in, height_in, args.max_width)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    raw = destination.with_name(destination.stem + "_raw.mp4")
    if raw == source:
        cap.release()
        raise ValueError("Input video conflicts with temporary output file.")

    writer = cv2.VideoWriter(
        str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {raw}")

    frames = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (width, height))
            prediction = model.predict(
                frame, device=0 if use_gpu else "cpu",
                imgsz=args.imgsz, conf=args.conf, verbose=False,
            )[0]

            detections = []
            for box in prediction.boxes:
                xyxy = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                detection = make_detection(
                    *xyxy, float(box.conf[0]), model.names[class_id]
                )
                if detection is not None:
                    detections.append(detection)

            tracks = tracker.update_tracks(detections, frame=frame)
            visible_count = 0
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue

                x1, y1, x2, y2 = map(int, track.to_ltrb())
                x1, x2 = max(0, x1), min(width - 1, x2)
                y1, y2 = max(0, y1), min(height - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                visible_count += 1
                label = f"{track.get_det_class()} ID:{track.track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                )

            frames += 1
            current_fps = frames / max(time.perf_counter() - started, 1e-9)
            cv2.rectangle(frame, (0, 0), (min(width, 330), 30), (0, 0, 0), -1)
            cv2.putText(
                frame, f"Tracked: {visible_count} | Process: {current_fps:.1f} FPS",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            writer.write(frame)
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()

    if frames == 0:
        raw.unlink(missing_ok=True)
        raise RuntimeError("No frames were decoded from the input video.")

    elapsed = time.perf_counter() - started
    # H.264 works well in browsers and Colab; if ffmpeg isn't installed,
    # preserve the original MPEG-4 encoded MP4 instead.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(raw),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination)
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            raw.unlink(missing_ok=True)
        else:
            print("Warning: H.264 conversion failed; keeping MP4V output.")
            print(completed.stderr[-500:])
            raw.replace(destination)
    else:
        print("Note: ffmpeg unavailable; output uses MP4V encoding.")
        raw.replace(destination)

    print("PROCESSING COMPLETE")
    print(f"Frames: {frames}")
    print(f"Processing FPS: {frames / max(elapsed, 1e-9):.2f}")
    print(f"Device: {'CUDA' if use_gpu else 'CPU'}")
    print(f"Output: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="outputs/tracking_output.mp4")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = entire video)")
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf must be in the range (0, 1].")
    if args.imgsz < 32 or args.max_width < 2 or args.max_age < 1 or args.max_frames < 0:
        parser.error("Invalid image size, max width, max age, or max frames.")
    process_video(args)


if __name__ == "__main__":
    main()
