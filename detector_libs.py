import cv2
import numpy as np


def draw_rectangle(frame, bbox):
    return cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)


def detect_blazeface(frame_rgb,blazeface):
    height, width, _ = frame_rgb.shape
    width_scale = width / 128
    height_scale = height / 128
    img = cv2.resize(frame_rgb, [128,128])
    detections = blazeface.predict_on_image(img)

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)
    
    for i in range(detections.shape[0]):
        ymin = int(detections[i, 0] * 128 * height_scale)
        xmin = int(detections[i, 1] * 128 * width_scale)
        ymax = int(detections[i, 2] * 128 * height_scale)
        xmax = int(detections[i, 3] * 128 * width_scale)
        frame_rgb = draw_rectangle(frame_rgb, [xmin,ymin,xmax,ymax])
    
    return frame_rgb


def detect_yolov8(frame_rgb, yolov8):
    detections = yolov8(frame_rgb,verbose=False)[0].boxes

    for detection in detections:
        xmin,ymin,xmax,ymax = [int(b) for b in detection.xyxy.cpu().numpy()[0]]
        frame_rgb = draw_rectangle(frame_rgb, [xmin,ymin,xmax,ymax])
    
    return frame_rgb


def detect_haarcascade(frame_rgb, haarcascade):
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    detections = haarcascade.detectMultiScale(img, 1.1, 4)

    for (xmin, ymin, w, h) in detections:
        xmax = xmin + w
        ymax = ymin + h
        frame_rgb = draw_rectangle(frame_rgb, [xmin,ymin,xmax,ymax])

    return frame_rgb


def detect_mtcnn(frame_rgb, mtcnn):
    detections = mtcnn(frame_rgb)
    
    for (xmin, ymin, xmax, ymax) in detections:
        frame_rgb = draw_rectangle(frame_rgb, [xmin,ymin,xmax,ymax])
    
    return frame_rgb
