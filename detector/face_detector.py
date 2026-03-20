# librerias
from ultralytics import YOLO
import cv2
import os

class faceDetector:
    def __init__(self, model_path=os.path.join("modelos", "yolov8n.pt")):
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileExistsError(f"El modelo{model_path} no se encuentra")
        return YOLO(model_path)
    
    def detect_face(self, frame):
        resultado = self.model(frame)
        faces = []
        
        for resultado in resultado:
            for box in resultado.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })
                
        return faces