import cv2
import torch
import pandas as pd
import os
import time
import sys
import select
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# Detect GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(f"Using device: {device}")

# Initialize models on the selected device
mtcnn = MTCNN(keep_all=False, device=device)  # Detect a single face
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

# Auto-load attendees from 'faces' folder
faces_folder = "faces"
attendees = {}

for file in os.listdir(faces_folder):
    if file.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):  # Allow multiple image formats
        name = os.path.splitext(file)[0]  # Extract name from filename
        attendees[name] = os.path.join(faces_folder, file)

print(f"Loaded attendees: {list(attendees.keys())}")

# Attendance tracking
attendance_marked = {name: "A" for name in attendees}  # Default "Absent"

# Get the current session timestamp for column name
session_time = time.strftime("%Y-%m-%d %H:%M:%S")

# Function to compute face embeddings
def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = mtcnn(img)  # Extract face
        if img is None:
            print(f"⚠️ No face detected in {img_path}, skipping...")
            return None

        img = img.to(device)  # Move to GPU
        
        # Ensure the tensor shape is correct
        if img.dim() == 3:
            img = img.unsqueeze(0)  # Convert to [1, 3, 160, 160] for batch processing
        
        # Pass through FaceNet model
        face_embedding = resnet(img).detach().cpu()  # Move back to CPU
        return face_embedding
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return None

# Store embeddings of known faces
stored_embeddings = {name: get_embedding(img_path) for name, img_path in attendees.items()}
stored_embeddings = {name: emb for name, emb in stored_embeddings.items() if emb is not None}

# Load or create an attendance sheet
attendance_file = "attendance.xlsx"
df = pd.read_excel(attendance_file, index_col=0) if os.path.exists(attendance_file) else pd.DataFrame(index=attendees.keys())

# Add new column for the current session
df[session_time] = "A"  # Default "A" (Absent)

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect faces in the frame
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)

            # Ensure coordinates are within frame bounds
            x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1]), min(h, frame.shape[0])

            # Crop the face
            face_image = frame[y:h, x:w]

            # Check if face_image is valid
            if face_image is None or face_image.size == 0:
                print("⚠️ Empty face image detected, skipping...")
                continue  # Skip this face

            # Convert to PIL Image
            face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

            # Get embedding for detected face
            face_tensor = mtcnn(face_pil)

            if face_tensor is None:
                print("⚠️ No face detected, skipping frame...")
                continue  # Skip processing this frame

            face_tensor = face_tensor.to(device)  # Move to GPU
            face_embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu()  # Move back to CPU

            # Compare with stored embeddings
            for name, stored_emb in stored_embeddings.items():
                similarity = torch.nn.functional.cosine_similarity(face_embedding, stored_emb)

                if similarity > 0.7:  # Threshold for recognition
                    if attendance_marked[name] == "A":  # Only mark once per session
                        attendance_marked[name] = "P"
                        df.loc[name, session_time] = "P"
                        print(f"✅ Attendance marked for {name}")

                    # Draw box and name on the frame
                    cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    break  # Exit loop after first match

    # Show the webcam feed
    cv2.imshow('Face Recognition Attendance System (PyTorch)', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        if sys.stdin.read(1) == 'q':
            break

cap.release()
cv2.destroyAllWindows()

# Save attendance to Excel
df.to_excel(attendance_file)
print(f"✅ Attendance saved to {attendance_file}")

# ✅ Upload to Google Sheets
def upload_to_google_sheets(df):
    try:
        # Google Sheets API Setup
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("google_sheets_credentials.json", scope)
        client = gspread.authorize(creds)

        # Open Google Sheet
        sheet = client.open("AttendanceSheet").sheet1
        
        df = df.fillna("")

        # Convert DataFrame to List of Lists
        data = [df.columns.values.tolist()] + df.reset_index().values.tolist()

        # Update Google Sheet
        sheet.clear()  # Clear existing data
        sheet.update(data)  # Upload new data

        print("✅ Attendance uploaded to Google Sheets!")
    except Exception as e:
        print(f"❌ Error uploading to Google Sheets: {e}")

upload_to_google_sheets(df)
