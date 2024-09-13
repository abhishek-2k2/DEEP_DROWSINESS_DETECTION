<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
  
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 0; width: 80%; margin: 0 auto; padding: 20px;">
    <h1 style="color: #333;">Drowsiness Detection System</h1>
    <p>This project demonstrates a drowsiness detection system using YOLOv5 for object detection and custom training. The system captures real-time video from a webcam and detects whether a person is awake or drowsy.</p>

    <h2 style="color: #555;">Features</h2>
    <ul style="margin: 0; padding: 0 0 0 20px;">
        <li>Real-time drowsiness detection using YOLOv5 and custom-trained models.</li>
        <li>Image collection from a webcam to create a custom dataset for training.</li>
        <li>Integration with OpenCV for capturing video and rendering detection results.</li>
    </ul>

    <h2 style="color: #555;">Installation</h2>
    <pre style="background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd;">
pip install torch torchvision torchaudio
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
    </pre>

    <h2 style="color: #555;">Training the Model</h2>
    <pre style="background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd;">
python train.py --img 320 --batch 16 --epochs 500 --data dataset.yml --weights yolov5s.pt --workers 2
    </pre>

    <h2 style="color: #555;">Running the Detection</h2>
    <pre style="background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd;">
import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    </pre>
</body>
</html>
