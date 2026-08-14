# YOLOv12 Object Detection & Privacy Blurring

Three small OpenCV + Ultralytics scripts: object detection on a still image, detection on video with FPS display, and a privacy module that blurs detected people frame by frame.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3.195-green)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-orange)

<img src="https://github.com/user-attachments/assets/a1f83abc-e5fc-4ccf-a283-03e460873e9b" width="700"/>

---

## What it does

- **Image detection** — runs a pretrained YOLOv12 model on a single image and draws boxes and labels for the 80 COCO classes.
- **Video detection** — processes a video file frame by frame, overlays boxes and a live FPS counter, and writes the annotated result to MP4.
- **Privacy blurring** — detects people and blurs their bounding boxes before drawing any overlays, so overlapping detections can't leave a subject visible. A single `BLUR_CLASSES` constant switches between "people only" and "everything detected".

This project uses **pretrained weights only**. There is no training step, no custom dataset, and no train/validation/test split — the models ship already trained on COCO.

---

## Tech stack

| Component       | Tool                    |
|-----------------|-------------------------|
| Language        | Python 3.10             |
| Detection       | Ultralytics 8.3.195     |
| Computer vision | OpenCV (cv2)            |

---

## Project structure

```
├── main.py                # Image inference with custom box and label rendering
├── main_video.py          # Video file detection with FPS overlay and MP4 output
└── privacy_blurring.py    # Person detection with two-pass blur-then-draw rendering
```

---

## Configuration

| Parameter      | Value | Notes                                                                 |
|----------------|-------|-----------------------------------------------------------------------|
| `conf`         | 0.25  | Minimum confidence. Lower values catch distant subjects but add noise. |
| `iou`          | 0.7   | NMS threshold. Ultralytics default — lower values suppress genuine overlapping detections in crowds. |
| `blur_ratio`   | 50    | Box blur kernel size for the privacy module.                          |
| `BLUR_CLASSES` | `[0]` | COCO class IDs to blur. `[0]` is person; `None` blurs everything detected. |

`main.py` uses `yolo12s.pt` for better accuracy on small and partially occluded subjects. The video scripts use `yolo12n.pt` to stay usable on CPU.

---

## Known limitations

These are properties of the approach, not bugs to be tuned away:

- **Box blur is not face anonymization.** YOLO returns a full-body box, so the blur covers the whole person rather than targeting the face. A dedicated face detector would be needed for face-level anonymization.
- **Missed frames leave subjects visible.** Detection runs independently per frame. If the model misses a person in one frame, that frame shows them unblurred — and a single frame is enough. Object tracking would help by carrying detections across gaps.
- **False positives on person-shaped objects.** Tall, high-contrast vertical objects (road barriers, signage) are intermittently detected as people. A larger model reduces this; the nano model does not.
- **No quantitative evaluation.** Thresholds were chosen by visual inspection on sample media, not measured against a labelled evaluation set. There is no mAP figure for this configuration.
- **CPU throughput is modest.** Roughly 11 FPS with `yolo12n` at source resolution on CPU. Larger models drop this to single digits; lowering `imgsz` is the main lever.

---

## Getting started

```bash
git clone https://github.com/EfeNayin/yolov12-detection.git
cd yolov12-detection

pip install -r requirements.txt

python main.py               # image detection
python main_video.py         # video detection
python privacy_blurring.py   # privacy blurring
```

Model weights download automatically on first run. Press `q` to stop either video script early.