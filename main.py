import cv2
import argparse
from datetime import datetime
from detector.face_detector import faceDetector
from detector.face_recognizer import faceRecognizer
from detector.utils import draw_bbox_with_label
import os

def main():
    parser = argparse.ArgumentParser(description='Sistema de asistencia por reconocimiento facial')
    parser.add_argument('--mode', choices=['detect', 'register'], default='detect', 
                        help='Modo: detect (reconocimiento) o register (registrar nuevos rostros)')
    parser.add_argument('--name', type=str, help='Ingrese el nombre a registrar')
    args = parser.parse_args()
    
    detector = faceDetector()
    recognizer = faceRecognizer()
    
    cap = cv2.VideoCapture(0)
    
    if args.mode == 'register' and not args.name:
        print("Error: Se debe proporcionar un --name en modo registro")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = detector.detect_face(frame)
            
            if args.mode == 'detect':
                for face in faces:
                    x1, y1, x2, y2 = face['bbox']  
                    cropped_face = frame[y1:y2,x1:x2]
                    
                    name, confidence = recognizer.recognize(cropped_face)
                    
                    if confidence > 0.6:
                        cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 255, 0),2)
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),2)
                        
                        register_attendance(name)
            else:
                if faces:
                    face = faces[0]
                    x1, y1, x2, y2 = face['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0),2)
                    cv2.putText(frame, "presiona 's' para guardar", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),2)
            cv2.imshow("Sistema por reconocimiento facial", frame)
            
            key = cv2.waitKey(1)
            if key == ord ('q'):
                break
            elif key == ord('s') and args.mode == 'registrer' and faces:
                # Se guarda un nuevo rostro
                face_img = frame[y1:y2, x1:x2]
                recognizer.register_face(face_img, args.name)
                print(f"Rostro de {args.name} se regsitro recientemente")
                break
    finally: 
        cap.release()
        cv2.destroyAllWindows()
        
def register_attendance(name):
    database_dir = "database"
    csv_path = os.path.join(database_dir, "registro.csv")
    os.makedirs(database_dir, exist_ok=True)
    
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{name}, {timestamp}\n")
    
    print(f"Se registro asistencia de: {name} a las ({timestamp})")

if __name__ == "__main__":
    main()