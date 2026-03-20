import cv2
import os
from datetime import datetime

def preprocess_face(face_img, target_size=(100,100)):
    gray = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray,target_size)
    normalized = resized /255.0
    return normalized

def save_attendance_record(name, filename='registros.cvs'):
    # Guarda un registro de asistencia con timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'database/{filename}','a') as f:
        f.write(f"{name},{timestamp}\n")

def draw_bbox_with_label(frame, bbox, label, color=(0,255,0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1),(x2,y2), color, 2)
    
    # Fondos texto
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.rectangle(frame, (x1,y1 - text_height -10), (x1 + text_width, y1), color, -1)
    
    # Texto
    cv2.putText(frame, label, (x1,y1 -5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
    
    return frame