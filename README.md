# FaceDetectZoo

Face recognition begins with face detection. 

Towards developing attacks against face recognition systems, this idea has flourished. 

A simple webcam interface that embeds multiple face detection modules & algorithms. 

With small tweaks, this interface can be used to test the transferability of an adversarialy\
transformed camera feed, trained on a specific system, against other face detection algorithms.

The goal is to embed many more available detectors.

## Currently supported detectors
- [MTCNN](https://link.springer.com/chapter/10.1007/978-981-13-6508-9_8)
- [BlazeFace](https://arxiv.org/abs/1907.05047)
- [Haar Cascades](https://ieeexplore.ieee.org/abstract/document/8525935?casa_token=6wtGKjeyKnUAAAAA:1Hn_jjbOUxir4Qr34qshJo0uKvqVC94tn4i8vDohATsgFcGrX02SAFdYOSmQFpeuRVNa6qQIRwY)
- [YOLOv8n-face](https://github.com/akanametov/yolov8-face)

## Requirements

Create a venv and install dependencies via ` pip install -r requirements.txt`

### MTCNN

MTCNN implementation is available from `facenet_pytorch` module.

Credit: [timesler](https://github.com/timesler/facenet-pytorch)
### BlazeFace

To use BlazeFace: 

1. Download [`anchors.npy`](https://github.com/hollance/BlazeFace-PyTorch/blob/master/anchors.npy) & [`blazeface.pth`](https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.pth)
2. Paste inside `./detectors/blazeface/`

Credit: [hollance](https://github.com/hollance/BlazeFace-PyTorch)

### Haar Cascades
To use Haar Cascades:

1. Download [`haarcascade_frontalface_default.xml`](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
2. Paste inside `./detectors/haarcascade/`

### YOLOv8n-face
To use YOLOv8n-face:

1. Download [`yolov8n-face.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt)
2. Paste inside `./detectors/yolov8n-face/`
