import cv2
import math
from ultralytics import YOLO

image_path = "Images/people.jpg"
image = cv2.imread(image_path)


model = YOLO("yolo12s.pt")

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


results = model.predict(image, conf=0.25, iou=0.7)

print("Number of boxes found:", len(results[0].boxes))

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1

for result in results:
    boxes = result.boxes
    for box in boxes:

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        classNameInt = int(box.cls[0])
        className = cocoClassNames[classNameInt]
        conf = math.ceil(box.conf[0] * 100) / 100

        label = f"{className}: {conf}"

        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        if y1 - text_h - baseline >= 0:
            label_y = y1
        else:
            label_y = y2 + text_h + baseline

        cv2.rectangle(image, (x1, label_y - text_h - baseline), (x1 + text_w, label_y), (255, 0, 0), -1)
        cv2.putText(image, label, (x1, label_y - baseline), FONT, FONT_SCALE, (255, 255, 255),
                    FONT_THICKNESS, cv2.LINE_AA)

cv2.imshow("YOLOv12 - Image Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()