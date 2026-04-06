import cv2
import math
import time
from ultralytics import YOLO

video_path = "Videos/riding_bicycle.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_rate = int(cap.get(cv2.CAP_PROP_FPS))

output_video = cv2.VideoWriter('privacy_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_rate, (frame_width, frame_height))

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


blur_ratio = 50
prev_time = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(frame, conf=0.15, iou=0.1, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
       
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            roi = frame[y1:y2, x1:x2]
      
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                blurred_roi = cv2.blur(roi, (blur_ratio, blur_ratio))
                frame[y1:y2, x1:x2] = blurred_roi


            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            class_id = int(box.cls[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            label = f"{cocoClassNames[class_id]}: {conf}"
            
            (tw, th), _ = cv2.getTextSize(label, 0, 0.5, 2)
            cv2.rectangle(frame, (x1, y1), (x1 + tw, y1 - th - 5), (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 0, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)


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