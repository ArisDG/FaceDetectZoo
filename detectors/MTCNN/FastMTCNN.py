from facenet_pytorch import MTCNN
import cv2

class FastMTCNN(object):
  def __init__(self, *args, **kwargs):
    self.mtcnn = MTCNN(*args, **kwargs)

  def __call__(self, frame):
    boxes, _ = self.mtcnn.detect(frame)
    detections = []

    for box in boxes:
      box = [int(b) for b in box]
      detections.append(box)

    return detections
