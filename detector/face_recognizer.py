import cv2
import os
import numpy as np
from datetime import datetime
from .utils import preprocess_face, save_attendance_record

class faceRecognizer:
    def __init__(self):
        self.know_faces = []
        self.know_names = []
        self.load_know_faces()
        
    def load_know_faces(self):
        if not os.path.exists('database/rostros'):
            os.makedirs('database/rostros')
        
        for filename in os.listdir('database/rostros'):
            if filename.endswith('.jpg'):
                names = filename.split('.')[0]
                img_path = os.path.join('database/rostros', filename)
                img = cv2.imread(img_path)
                self.know_faces.append(self.preprocess_face(img))
                self.know_names.append(names)
                
    def preprocess_face(self, face_img):
        if not self.know_faces:
            return 'Desconocido rostro', 0.0
    
    def recognize(self, face_img):
        if not self.know_faces:
            return 'Rostro no reconocido', 0.0
        processed = self.preprocess_face(face_img)
        
        similarities = []
        for know_faces in self.know_faces:
            res = cv2.matchTemplate(processed, know_faces, cv2.TM_CCOEFF_NORMED)
            similarities.append(res[0][0])
            
            max_idx = np.argmax(similarities)
            max_similarities = similarities[max_idx]
            
            if max_similarities > 0.6:
                return self.know_names[max_idx], max_similarities
            else:
                return 'Desconocido rostro', max_similarities
    
    def register_face(self, face_img, name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}.jpg"
        cv2.imwrite(f'database/rostros/{filename}', face_img)
        self.load_know_faces()