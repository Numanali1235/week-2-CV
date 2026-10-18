"""Run a pretrained YOLO model on a video, keep only 'person', draw boxes + confidence, save output."""
import argparse
import cv2
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to input video")
    p.add_argument("--output", default="output.mp4", help="path to output video")
    p.add_argument("--model", default="yolov8n.pt", help="pretrained weights (auto-downloaded)")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    args = p.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # classes=[0] -> COCO class 0 = person only
        result = model.predict(frame, classes=[0], conf=args.conf, verbose=False)[0]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"person {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        out.write(frame)
        frames += 1

    cap.release()
    out.release()
    print(f"Done. {frames} frames -> {args.output}")


if __name__ == "__main__":
    main()
