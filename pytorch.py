import cv2
import torch
import pandas as pd
import os
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=False, device='cpu')

# Initialize FaceNet model for face recognition
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Define attendees with stored face embeddings
attendees = {
    "Steve": "faces/512px-Steve_Jobs_Headshot_2010-CROP_(cropped_2).webp",
    "Tesla": "faces/512px-Tesla_circa_1890.webp",
    "Harshil": "faces/file.enc",
    "Person4": "faces/ratantata",
}

attendance_data = {}

# Function to compute face embeddings
def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = mtcnn(img)  # Extract face
        if img is not None:
            return resnet(img.unsqueeze(0)).detach()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    return None

# Store embeddings of known faces
stored_embeddings = {}
for name, img_path in attendees.items():
    emb = get_embedding(img_path)
    if emb is not None:
        stored_embeddings[name] = emb

# Create or load an attendance log
attendance_log_path = 'attendance.csv'
if os.path.exists(attendance_log_path):
    attendance_data = pd.read_csv(attendance_log_path, index_col=0).to_dict()['Timestamp']
else:
    attendance_data = {}

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect faces in the current frame
    boxes, _ = mtcnn.detect(frame)
    
    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)
            face_image = frame[y:h, x:w]
            face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

            # Get embedding for detected face
            face_tensor = mtcnn(face_pil)
            if face_tensor is None:
                continue  # Skip if no face detected

            face_embedding = resnet(face_tensor.unsqueeze(0)).detach()
            
            # Compare embeddings
            for name, stored_emb in stored_embeddings.items():
                similarity = torch.nn.functional.cosine_similarity(face_embedding, stored_emb)
                
                if similarity.item() > 0.7:  # Threshold for recognition
                    if name not in attendance_data:
                        attendance_data[name] = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"Attendance marked for {name} at {attendance_data[name]}")

                        # Save attendance log
                        df = pd.DataFrame(list(attendance_data.items()), columns=["Name", "Timestamp"])
                        df.to_csv(attendance_log_path, index=False)

                    # Draw box and name on the frame
                    cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    break

    # Show the webcam feed
    cv2.imshow('Face Recognition Attendance System (PyTorch)', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
