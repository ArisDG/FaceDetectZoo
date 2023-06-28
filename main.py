import cv2
import tkinter as tk
from PIL import Image, ImageTk
import detector_libs
import torch
from ultralytics import YOLO
from detectors.blazeface.blazeface import BlazeFace as BlazeFace_
from detectors.MTCNN.FastMTCNN import FastMTCNN




def update_memory():
    detector_list =['BlazeFace','yolov8n-face','Haar_cascade','MTCNN']
    for x in detector_list:
        if x in globals():
            del globals()[x]

    if globals()['cur_detector'] == 'BlazeFace':     
        gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        blazeface = BlazeFace_().to(gpu)
        blazeface.load_weights("./detectors/blazeface/blazeface.pth")
        blazeface.load_anchors("./detectors/blazeface/anchors.npy")
        blazeface.eval()
        blazeface.min_score_thresh = 0.75
        blazeface.min_suppression_threshold = 0.3
        dynamic_global_variable('BlazeFace',blazeface)

    if globals()['cur_detector'] == 'yolov8n-face':
        yolov8n = YOLO("./detectors/yolov8n-face/yolov8n-face.pt")
        dynamic_global_variable('yolov8n-face',yolov8n)

    if globals()['cur_detector'] == 'Haar_cascade':
        haar_cascade = cv2.CascadeClassifier('./detectors/haarcascade/haarcascade_frontalface_default.xml')
        dynamic_global_variable('Haar_cascade',haar_cascade)     

    if globals()['cur_detector'] == 'MTCNN':
        gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mtcnn = FastMTCNN(keep_all=True,device=gpu)
        dynamic_global_variable('MTCNN',mtcnn)     


def dynamic_global_variable(name, value):
    globals()[name] = value


def detect_face(frame_rgb):
    if globals()['cur_detector'] == 'BlazeFace':
        frame_rgb = detector_libs.detect_blazeface(frame_rgb, globals()['BlazeFace'])

    if globals()['cur_detector'] == 'yolov8n-face':
        frame_rgb = detector_libs.detect_yolov8(frame_rgb, globals()['yolov8n-face'])
    
    if globals()['cur_detector'] == 'Haar_cascade':
        frame_rgb = detector_libs.detect_haarcascade(frame_rgb, globals()['Haar_cascade'])

    if globals()['cur_detector'] == 'MTCNN':
        frame_rgb = detector_libs.detect_mtcnn(frame_rgb, globals()['MTCNN'])

    return frame_rgb


def handle_selection(*args):
    dynamic_global_variable('cur_detector', dropdown.get())
    update_memory()


def update_frame():
    _, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_detected = detect_face(frame_rgb)
    frame_resized = cv2.resize(frame_detected, (640, 480))
    img = Image.fromarray(frame_resized)
    img_tk = ImageTk.PhotoImage(image=img)
    panel.config(image=img_tk)
    panel.image = img_tk
    panel.after(1, update_frame)


#==================================================================================================================================#

if __name__ == '__main__':
    dynamic_global_variable('cur_detector', 'None')
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam index
    window = tk.Tk()
    window.title("Webcam GUI")
    options = ['None', 'BlazeFace','yolov8n-face','Haar_cascade','MTCNN']
    dropdown = tk.StringVar(window)
    dropdown.set(options[0])
    dropdown_menu = tk.OptionMenu(window, dropdown, *options)
    dropdown_menu.pack()
    panel = tk.Label(window)
    panel.pack()
    dropdown.trace("w", handle_selection)
    update_frame()
    window.mainloop()
    cap.release()



