# Video-Based Person Identification with YOLOv8

This project implements an automated system for detecting people in video recordings. It uses pretrained YOLOv8 weights to locate humans in each frame, displays a bounding box around every detection, and shows the associated confidence score. All other object classes are ignored.

## Installation Instructions

Before running the application, install the necessary Python packages:

```bash
pip install -r requirements.txt
```

## How to Run

Use the following command to process a video:

```bash
python detect_people.py --input input.mp4 --output output.mp4 --conf 0.4
```

## Command-Line Settings

The application provides several options to customize the detection process:

- **Input video (`--input`):** Specifies the video file to analyze.
- **Output file (`--output`):** Sets the name of the generated video.
- **Model (`--model`):** Defines the YOLO weights to use. By default, the program uses `yolov8n.pt`.
- **Confidence (`--conf`):** Determines the minimum confidence required to accept a detection. The default value is `0.4`.

## Generated Results

Once execution finishes, the annotated footage is stored in `output.mp4`. Each detected person is marked with a rectangular outline and a confidence score, making it easier to observe the detection results frame by frame.