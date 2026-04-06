import cv2
import math
from ultralytics import YOLO

image_path = "Images/people.jpg"
image = cv2.imread(image_path)


model = YOLO("yolo12n.pt")

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


results = model.predict(image, conf=0.15, iou=0.1)


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

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.5, 2)[0]
        c2 = x1 + text_size[0], y1 - text_size[1] - 3
        cv2.rectangle(image, (x1, y1), c2, (255, 0, 0), -1)
        cv2.putText(image, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

cv2.imshow("YOLOv12 - Image Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()