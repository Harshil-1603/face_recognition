import cv2
import pandas as pd
from deepface import DeepFace
import os
import time

# Define the people you want to store in the system
attendees = ["Harshil", "bhuvnesh"]  # Replace with your names
attendance_data = {}

# Create or load an existing attendance log
attendance_log_path = 'attendance.csv'
if os.path.exists(attendance_log_path):
    attendance_data = pd.read_csv(attendance_log_path, index_col=0).to_dict()['Name']
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
    result = DeepFace.analyze(frame, actions=['face_detection'], enforce_detection=False)

    # Check if faces are detected
    for face in result[0]['instances']:
        x, y, w, h = face['box']
        face_image = frame[y:y+h, x:x+w]

        # Try to recognize the face
        try:
            recognized_person = DeepFace.detectFace(face_image, db_path="/home/harshil/Desktop/ED")
            if recognized_person and recognized_person[0]['identity'] in attendees:
                name = recognized_person[0]['identity']
                if name not in attendance_data:
                    attendance_data[name] = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Attendance marked for {name} at {attendance_data[name]}")
                    # Save attendance log
                    df = pd.DataFrame(list(attendance_data.items()), columns=["Name", "Timestamp"])
                    df.to_csv(attendance_log_path, index=False)
        except Exception as e:
            print(f"Error recognizing face: {e}")
    
    # Display the frame with a rectangle around the detected face
    cv2.imshow('Face Recognition Attendance System', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
