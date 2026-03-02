#-------------Steps-------------#

#read an Image using OpenCV
#Load the YOLOv12 Model and Perform Object Detection
#add the Confidence Value "conf" ----> detect edebilmek için minimum confidence değeri ayarlama
#Add NMS IOU 'iou'------> detect edilmiş bir belirli bir seviyeden sonra tekrar detect etmesini sınırlayacı bir fonksiyon.
# classes = [0,1] -->sadece insanları ve bisikletleri detect etmek istediğim için böyle bir şey yaptım.
# max_det = 1 ---> maximum 1 kişiyi detect ettik
#show = True ----> Modelin çıktısını (örneğin, bir görüntü veya video akışı) anında görmek için.
#save_txt = True ----> her bir detect in koordinatlarını çıkarır.
#save_crop -----> Algılamaların kırpılmış görüntülerini kaydeder.
#

#----------------------------------#

#Import All the Required Libraries

import cv2
from ultralytics import YOLO
import math

#Read an Image and image using OpenCV
image = cv2.imread("Images/people.jpg")
  #cap = cv2.VideoCapture(r"Videos/riding_bicycle.mp4")
#Load Yolov12 Model
model = YOLO("yolo12n.pt")
#Classes in the MS COCO data
cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]  # bunu yazmana gerek yok sadece örnek olsun diye

#Object Detection using YOLOv12
results = model.predict(image,conf = 0.15,iou= 0.1) # bununla tek başına hem detect hem de yazı yazabilirsin ama kendi stlinin için bunu da yapabilirsin.
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1,y1,x2,y2 = box.xyxy[0]
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(image,(x1,y1),(x2,y2), (255,0,0), 2)
        classNameInt = int(box.cls[0])
        className = cocoClassNames[classNameInt]
        conf = math.ceil(box.conf[0]*100) / 100
        label = className + ":" + str(conf)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.5, 2)[0]
        c2 = x1 + text_size[0],y1 - text_size[1] - 3
        cv2.rectangle(image,(x1,y1),c2,(255,0,0), -1)
        cv2.putText(image,label,(x1,y1 - 2),0,0.5,(255,255,255),thickness=1,lineType=cv2.LINE_AA)

#Display the image using OpenCV
cv2.imshow("Image",image)
cv2.waitKey(0)

