# YOLOv12 Real-Time Object Detection & Privacy Pipeline

A high-performance computer vision pipeline using state-of-the-art YOLOv12. Covers the full workflow — from static image inference to real-time video processing and automated data anonymization.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv12](https://img.shields.io/badge/Framework-YOLOv12-green)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-orange)

---

## 🚀 Key Features

- **SOTA Object Detection** — Leverages YOLOv12 for ultra-fast and accurate object localization across 80+ COCO classes.
- **Real-Time Video Analytics** — Optimized frame-by-frame processing with dynamic FPS monitoring and multi-format output writing (MP4v).
- **Privacy-First Design** — Automated Region of Interest (ROI) blurring for data privacy and GDPR compliance.
- **Modular Architecture** — Clean, refactored codebase following production-ready industry standards.

---

## 🛠 Tech Stack

| Component       | Tool                  |
|-----------------|-----------------------|
| Language        | Python 3.10           |
| Framework       | Ultralytics YOLOv12   |
| Computer Vision | OpenCV (cv2)          |
| Utilities       | math, time            |

---

## 📂 Project Structure

```
├── image_detection.py     # Static image inference with custom bounding box rendering
├── video_detection.py     # Real-time detection on video files / webcam with FPS tracking
└── privacy_blurring.py    # Automated object masking and privacy protection module
```

---

## ⚙️ Configuration & Parameters

The pipeline is highly tunable for different use cases:

| Parameter      | Value | Description                                          |
|----------------|-------|------------------------------------------------------|
| `conf`         | 0.15  | Minimum confidence threshold for detections.         |
| `iou`          | 0.1   | NMS threshold to suppress redundant bounding boxes.  |
| `blur_ratio`   | 50    | Adjustable blurring intensity for the privacy module.|

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/EfeNayin/yolov12-detection.git
cd yolov12-detection

# Install dependencies
pip install ultralytics opencv-python

# Run object detection on an image
python image_detection.py

# Run real-time video detection
python video_detection.py

# Run privacy blurring module
python privacy_blurring.py
```
