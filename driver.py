import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import *
from darknet import Darknet

cfg_file = './cfg/yolov3.cfg'
weight_file = './weights/yolov3.weights'
namesfile = 'data/coco.names'
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(original_image, (m.width, m.height))
    iou_thresh = 0.4
    nms_thresh = 0.6
#    nms_thresh = 0.1

    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)

    for box in boxes:
        width = resized_image.shape[1]
        height = resized_image.shape[0]
        cls_conf = box[5]
        cls_id = box[6]
        x1 = int(np.around((box[0] - box[2]/2.0) * width))
        y1 = int(np.around((box[1] - box[3]/2.0) * height))
        x2 = int(np.around((box[0] + box[2]/2.0) * width))
        y2 = int(np.around((box[1] + box[3]/2.0) * height))
        
        resized_image = cv2.rectangle(resized_image, (x1,y1), (x2,y2), (255, 255, 0), 2)
        # conf_tx = class_names[cls_id]
        # resized_image = cv2.putText(resized_image, conf_tx, (1,100), cv2.FONT_HERSHEY_SIMPLEX, 4, 
        #     (255,255,255), 2, cv2.LINE_AA, False)  
    
    print_objects(boxes, class_names)
    cv2.imshow('frame',resized_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
