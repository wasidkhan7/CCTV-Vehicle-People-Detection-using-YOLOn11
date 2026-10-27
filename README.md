# 🚦 AI-Powered CCTV Vehicle & People Detection

A computer vision application that detects people and vehicles in video footage, tracks objects across frames using ByteTrack, and counts objects crossing configurable lines. Built with YOLO11, PyTorch, OpenCV, and Streamlit.

## 🚀 Project Overview

This project demonstrates how object detection and multi-object tracking can be integrated into an interactive video analytics application.

Users can upload a CCTV or traffic video, run the trained model, visualize detections and tracking IDs, view counting results, and download the processed video.

## ✨ Features

- **Person Detection:** Detect pedestrians in video footage.
- **Vehicle Detection:** Detect vehicles using a custom-trained YOLO11 model.
- **Object Tracking:** Track detected objects across video frames using ByteTrack.
- **Line-Crossing Counting:** Count tracked people and vehicles crossing configurable horizontal or vertical lines.
- **Video Upload:** Process user-uploaded video files.
- **Annotated Video:** Display and download the processed video.
- **Interactive Dashboard:** Adjust confidence thresholds and counting-line positions.
- **CPU and GPU Inference:** Supports CPU inference and automatically uses CUDA when available.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| YOLO11n | Object detection |
| PyTorch | Model inference |
| Ultralytics | YOLO and ByteTrack integration |
| OpenCV | Video processing and annotation |
| ByteTrack | Multi-object tracking |
| Streamlit | Interactive web interface |
| FFmpeg | H.264 video conversion |
| Git and GitHub | Version control and source hosting |

## 🧠 Model and Dataset

The detector was fine-tuned using an annotated person-and-vehicle dataset.

- **Model:** YOLO11n
- **Classes:** Person and vehicle
- **Training images:** 1,600
- **Validation images:** 400
- **Total images:** 2,000
- **Checkpoint:** `models/best.pt`

### Validation Results

The following metrics were obtained during evaluation on the 400-image validation set.

| Metric | Result |
|---|---:|
| Precision | 85.68% |
| Recall | 80.91% |
| mAP@50 | 87.88% |
| mAP@50–95 | 64.66% |

These results describe object-detection performance on the validation dataset. They do not directly measure tracking accuracy or counting accuracy.

## 🏗️ Application Workflow

1. The user uploads a video.
2. OpenCV reads the video frame by frame.
3. YOLO detects people and vehicles in each frame.
4. ByteTrack associates detections across frames and assigns tracking IDs.
5. The application checks whether tracked objects cross the configured counting lines.
6. OpenCV annotates the frames with bounding boxes, labels, tracking IDs, and counters.
7. FFmpeg converts the processed video to H.264 for browser playback.
8. Streamlit displays the results and provides a download button.

## 📂 Project Structure

```text
CCTV_VEHICLE_PEOPLE_DETECTION/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── best.pt
├── screenshots/
├── sample_videos/
└── outputs/
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CCTV_VEHICLE_PEOPLE_DETECTION
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

FFmpeg must also be installed and available on your system path.

On Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local address displayed in the terminal, usually `http://localhost:8501`.

## 🖥️ Hardware

The application was developed and tested locally on a system with:

- NVIDIA GeForce RTX 3060 Laptop GPU
- CUDA-enabled PyTorch
- Ubuntu Linux

CPU inference was also tested successfully.

Performance will vary depending on the hardware, video resolution, frame rate, and video length.

## 📸 Screenshots

Screenshots of the application are available in the `screenshots/` directory.

<!-- Add screenshots here after publishing the repository. -->

## ⚠️ Limitations

- Counting accuracy depends on detection quality, tracking consistency, camera angle, and counting-line placement.
- Tracking ID changes can cause inaccurate counts.
- Each tracked ID is counted at most once per class during a processing run.
- The current application processes uploaded videos rather than connecting directly to live RTSP camera streams.
- Detection metrics do not establish real-world counting accuracy.
- CPU processing may be slower than GPU processing.

## 🔮 Future Improvements

- RTSP live-camera integration.
- Direction-aware vehicle counting.
- Improved tracking and duplicate-count prevention.
- Traffic analytics dashboards and event history.
- Database integration for storing detection events.
- Docker-based deployment.
- Performance optimization for real-time inference.

## 👨‍💻 Author

**Wasid Khan**

Computer Science graduate interested in Computer Vision, Artificial Intelligence, and AI deployment.

---

If you find this project useful, consider starring the repository.