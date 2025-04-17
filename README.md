# Face Recognition Attendance System

A web-based interface for managing student attendance using face recognition technology.

## Features

- Add new students with photos (upload or webcam capture)
- Take attendance using face recognition
- Automatic attendance marking
- Google Sheets integration for attendance records

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `faces` directory in the project root:
```bash
mkdir faces
```

3. Set up Google Sheets API:
   - Create a Google Cloud Project
   - Enable Google Sheets API
   - Create a service account and download the credentials
   - Save the credentials as `google_sheets_credentials.json` in the project root
   - Share your Google Sheet with the service account email

4. Run the application:
```bash
python app.py
```

5. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

### Adding a New Student

1. Enter the student's name
2. Choose between uploading a photo or using the webcam
3. If using webcam:
   - Click "Capture Photo" when ready
   - Preview the captured photo
4. Click "Add Student" to save

### Taking Attendance

1. Click "Start Attendance" button
2. The system will open your webcam
3. Students should look at the camera
4. Attendance will be automatically marked
5. Press 'q' to quit the attendance session

## Notes

- Photos are automatically converted to WebP format for optimal storage
- Face recognition threshold is set to 0.7 (70% similarity)
- Attendance records are saved both locally and to Google Sheets
- The system requires a webcam for attendance taking 