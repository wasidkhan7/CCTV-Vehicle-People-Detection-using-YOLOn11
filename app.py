
from pathlib import Path
import tempfile
import subprocess
import cv2
import streamlit as st
import torch
from ultralytics import YOLO


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best.pt"

st.set_page_config(
    page_title="CCTV AI Analytics",
    page_icon="📹",
    layout="wide",
)

CLASS_NAMES = {
    0: "Person",
    1: "Vehicle",
}


# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


# --------------------------------------------------
# 3. VIDEO PROCESSING
# --------------------------------------------------

def process_video(
    model,
    input_path: str,
    output_path: str,
    confidence: float,
    line_mode: str,
    horizontal_position: float,
    vertical_position: float,
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video.")

    horizontal_y = int(height * horizontal_position)
    vertical_x = int(width * vertical_position)

    previous_positions = {}
    counted_vehicles = set()
    counted_people = set()

    vehicle_count = 0
    person_count = 0
    frame_number = 0

    progress = st.progress(0.0, text="Starting video processing...")
    status = st.empty()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=640,
                conf=confidence,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
            )[0]

            # Draw the selected counting lines.
            if line_mode in ("Horizontal", "Both"):
                cv2.line(
                    frame,
                    (0, horizontal_y),
                    (width, horizontal_y),
                    (0, 255, 255),
                    2,
                )

            if line_mode in ("Vertical", "Both"):
                cv2.line(
                    frame,
                    (vertical_x, 0),
                    (vertical_x, height),
                    (255, 0, 255),
                    2,
                )

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(
                    boxes, track_ids, class_ids
                ):
                    x1, y1, x2, y2 = map(int, box)

                    center_x = (x1 + x2) // 2
                    bottom_y = y2

                    if class_id == 0:
                        label = "Person"
                        color = (255, 100, 0)
                        counted_ids = counted_people
                    elif class_id == 1:
                        label = "Vehicle"
                        color = (0, 255, 0)
                        counted_ids = counted_vehicles
                    else:
                        continue

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    cv2.circle(
                        frame,
                        (center_x, bottom_y),
                        4,
                        (0, 0, 255),
                        -1,
                    )

                    cv2.putText(
                        frame,
                        f"{label} ID:{track_id}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                    object_key = (class_id, track_id)
                    previous_point = previous_positions.get(object_key)

                    if previous_point is not None:
                        previous_x, previous_y = previous_point

                        crossed_horizontal = (
                            line_mode in ("Horizontal", "Both")
                            and (
                                previous_y < horizontal_y <= bottom_y
                                or previous_y > horizontal_y >= bottom_y
                            )
                        )

                        crossed_vertical = (
                            line_mode in ("Vertical", "Both")
                            and (
                                previous_x < vertical_x <= center_x
                                or previous_x > vertical_x >= center_x
                            )
                        )

                        if (
                            crossed_horizontal or crossed_vertical
                        ) and track_id not in counted_ids:

                            counted_ids.add(track_id)

                            if class_id == 0:
                                person_count += 1
                            else:
                                vehicle_count += 1

                    previous_positions[object_key] = (
                        center_x,
                        bottom_y,
                    )

            # Add a readable count overlay.
            cv2.rectangle(
                frame,
                (10, 10),
                (390, 95),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                f"Vehicles counted: {vehicle_count}",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"People counted: {person_count}",
                (20, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 180, 0),
                2,
            )

            writer.write(frame)

            if total_frames > 0:
                fraction = min(frame_number / total_frames, 1.0)
                progress.progress(
                    fraction,
                    text=f"Processing frame {frame_number}/{total_frames}",
                )

            if frame_number % 10 == 0:
                status.write(
                    f"Processed {frame_number} frames"
                )

    finally:
        cap.release()
        writer.release()

    progress.progress(1.0, text="Processing complete!")

    return {
        "frames": frame_number,
        "vehicles": vehicle_count,
        "people": person_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


# --------------------------------------------------
# 4. STREAMLIT INTERFACE
# --------------------------------------------------

st.title("📹 AI-Powered CCTV Analytics")
st.markdown(
    """
    Detect and track **people and vehicles** in video using
    a fine-tuned YOLO11 model and ByteTrack.
    """
)

st.caption(
    "Computer Vision • Object Detection • Multi-Object Tracking"
)

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    line_mode = st.selectbox(
        "Counting line",
        ["Horizontal", "Vertical", "Both", "None"],
    )

    horizontal_position = st.slider(
        "Horizontal line position",
        min_value=0.10,
        max_value=0.90,
        value=0.55,
        step=0.05,
    )

    vertical_position = st.slider(
        "Vertical line position",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    st.divider()

    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    st.write(f"**Inference device:** {device_name}")

uploaded_video = st.file_uploader(
    "Upload a CCTV or traffic video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Start with a short video for faster processing.",
)

if uploaded_video is not None:
    st.video(uploaded_video, width=720)

    if st.button("🚀 Process Video", type="primary"):
        input_suffix = Path(uploaded_video.name).suffix or ".mp4"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = str(
                    Path(temp_dir) / f"input{input_suffix}"
                )

                intermediate_path = str(
                    Path(temp_dir) / "annotated_intermediate.mp4"
                )

                output_path = str(
                    Path(temp_dir) / "annotated_h264.mp4"
                )

                # Save the uploaded video temporarily.
                with open(input_path, "wb") as file:
                    file.write(uploaded_video.getbuffer())

                # Run YOLO detection, ByteTrack, and counting.
                with st.spinner(
                    "Running detection, tracking, and counting..."
                ):
                    summary = process_video(
                        model=model,
                        input_path=input_path,
                        output_path=intermediate_path,
                        confidence=confidence,
                        line_mode=line_mode,
                        horizontal_position=horizontal_position,
                        vertical_position=vertical_position,
                    )

                # Convert the video to browser-compatible H.264.
                with st.spinner(
                    "Converting video for browser playback..."
                ):
                    conversion = subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i", intermediate_path,
                            "-c:v", "libx264",
                            "-preset", "fast",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            "-an",
                            output_path,
                        ],
                        capture_output=True,
                        text=True,
                    )

                    if conversion.returncode != 0:
                        raise RuntimeError(
                            "FFmpeg conversion failed:\n"
                            + conversion.stderr[-2000:]
                        )

                # Read the converted video before the temporary
                # directory is deleted.
                output_bytes = Path(output_path).read_bytes()




            st.success("Video processing completed!")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Vehicles counted",
                summary["vehicles"],
            )

            col2.metric(
                "People counted",
                summary["people"],
            )

            col3.metric(
                "Frames processed",
                summary["frames"],
            )

            st.subheader("🎬 Annotated Video")
            st.video(output_bytes, width=720)

            st.download_button(
                "⬇️ Download Annotated Video",
                data=output_bytes,
                file_name="cctv_analytics_output.mp4",
                mime="video/mp4",
            )

            st.caption(
                f"Resolution: {summary['width']} × "
                f"{summary['height']} | "
                f"Input video FPS: {summary['fps']:.2f}"
            )

        except Exception as exc:
            st.error(f"Video processing failed: {exc}")

else:
    st.info("Upload a video to get started.")

st.divider()

st.markdown(
    """
    **Project:** CCTV Vehicle & People Detection

    **Model:** Fine-tuned YOLO11n  
    **Tracking:** ByteTrack  
    **Interface:** Streamlit
    """
)
