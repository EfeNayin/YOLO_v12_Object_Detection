import cv2
import time
from ultralytics import YOLO

video_path = "Videos/riding_bicycle.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_rate = int(cap.get(cv2.CAP_PROP_FPS))

output_video = cv2.VideoWriter('privacy_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_rate, (frame_width, frame_height))

model = YOLO("yolo12n.pt")

BLUR_CLASSES = [0]

cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1

blur_ratio = 50
prev_time = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.25, iou=0.7, classes=BLUR_CLASSES, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 > x1 and y2 > y1:
                detections.append((x1, y1, x2, y2, int(box.cls[0]), float(box.conf[0])))

    for x1, y1, x2, y2, _, _ in detections:
        frame[y1:y2, x1:x2] = cv2.blur(frame[y1:y2, x1:x2], (blur_ratio, blur_ratio))

    for x1, y1, x2, y2, class_id, raw_conf in detections:
        conf = round(raw_conf, 2)
        label = f"{cocoClassNames[class_id]}: {conf}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        if y1 - th - baseline >= 0:
            label_y = y1
        else:
            label_y = y2 + th + baseline

        cv2.rectangle(frame, (x1, label_y - th - baseline), (x1 + tw, label_y), (255, 0, 0), -1)
        cv2.putText(frame, label, (x1, label_y - baseline), FONT, FONT_SCALE, (255, 255, 255),
                    FONT_THICKNESS, cv2.LINE_AA)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    output_video.write(frame)
    cv2.imshow("Privacy Focused Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()