import cv2
import torch
import pandas as pd
import os
import time
import sys
import select
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# Detect GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models on the selected device
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

# Define attendees with stored face embeddings
attendees = {
    "Steve": "faces/512px-Steve_Jobs_Headshot_2010-CROP_(cropped_2).webp",
    "Tesla": "faces/512px-Tesla_circa_1890.webp",
    "Harshil": "faces/file.enc",
    "Person4": "faces/ratantata",
}

# Attendance tracking
attendance_marked = {name: "A" for name in attendees}  # Default everyone to "A"

# Get the current session timestamp for column name
session_time = time.strftime("%Y-%m-%d %H:%M:%S")

# Function to compute face embeddings
def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = mtcnn(img)  # Extract face
        if img is not None:
            img = img.to(device)  # Move image tensor to GPU
            return resnet(img.unsqueeze(0)).detach().cpu()  # Move back to CPU
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    return None

# Store embeddings of known faces
stored_embeddings = {name: get_embedding(img_path) for name, img_path in attendees.items()}
stored_embeddings = {k: v for k, v in stored_embeddings.items() if v is not None}  # Remove failed embeddings

# Load or create an attendance sheet
attendance_file = "attendance.xlsx"
if os.path.exists(attendance_file):
    df = pd.read_excel(attendance_file, index_col=0)
else:
    df = pd.DataFrame(index=attendees.keys())

# Add new column for the current session
df[session_time] = "A"  # Default to "Absent"

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
            x1, y1, x2, y2 = map(int, box)  # Correct box format
            face_image = frame[y1:y2, x1:x2]  # Crop face
            face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

            # Get embedding for detected face
            face_tensor = mtcnn(face_pil)

            if face_tensor is None:
                continue  # Skip if no face was extracted

            face_tensor = face_tensor.to(device)  # Move to GPU
            face_embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu()  # Move back to CPU

            # Compare embeddings
            for name, stored_emb in stored_embeddings.items():
                similarity = torch.nn.functional.cosine_similarity(face_embedding, stored_emb)

                if similarity > 0.7:  # Threshold for recognition
                    if attendance_marked[name] == "A":  # Only mark once
                        attendance_marked[name] = "P"
                        df.loc[name, session_time] = "P"
                        print(f"✅ Attendance marked for {name}")

                    # Draw box and name on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    break  # Exit loop after the first match

    # Show the webcam feed
    cv2.imshow('Face Recognition Attendance System (PyTorch)', frame)

    # Check for 'q' press in GUI or terminal
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        if sys.stdin.read(1) == 'q':
            break

cap.release()
cv2.destroyAllWindows()

# Save attendance to Excel
df.to_excel(attendance_file)
print(f"📂 Attendance saved to {attendance_file}")
