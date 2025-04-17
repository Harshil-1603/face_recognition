from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import subprocess
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'faces'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure the faces directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Global variable to track camera status
camera_in_use = False
camera_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_camera')
def check_camera():
    global camera_in_use
    with camera_lock:
        return jsonify({'in_use': camera_in_use})

@app.route('/release_camera')
def release_camera():
    global camera_in_use
    with camera_lock:
        camera_in_use = False
        # Release any existing camera connections
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            time.sleep(1)  # Give the system time to properly release the camera
        return jsonify({'success': True})

@app.route('/add_student', methods=['POST'])
def add_student():
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo uploaded'}), 400
    
    file = request.files['photo']
    name = request.form.get('name')
    
    if not name:
        return jsonify({'error': 'Name is required'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Convert to webp format
        filename = secure_filename(f"{name}.webp")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Read the image
        img = Image.open(file)
        # Save as webp
        img.save(filepath, 'WEBP')
        
        return jsonify({'success': True, 'message': f'Student {name} added successfully'})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    global camera_in_use
    with camera_lock:
        if camera_in_use:
            return jsonify({'error': 'Camera is currently in use by another process'}), 400
        
        try:
            camera_in_use = True
            # Release any existing camera connections
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
                time.sleep(1)  # Give the system time to properly release the camera
            
            # Run the face recognition script
            subprocess.run(['python', 'auto.py'])
            camera_in_use = False
            return jsonify({'success': True, 'message': 'Attendance taken successfully'})
        except Exception as e:
            camera_in_use = False
            return jsonify({'error': str(e)}), 500

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    try:
        # Get the base64 image data
        image_data = request.json.get('image')
        name = request.json.get('name')
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        # Convert base64 to image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save as webp
        filename = secure_filename(f"{name}.webp")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath, 'WEBP')
        
        return jsonify({'success': True, 'message': f'Student {name} added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 