from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import sys

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pose_utils import normalize_keypoints, calculate_similarity, detect_hand_activity

app = Flask(__name__)

# Constants
REFERENCE_IMAGE_PATH = 'reference_pose.png'
MODEL_PATH = 'pose_landmarker_lite.task'
SIMILARITY_THRESHOLD = 0.92

# State dictionary for shared data between threads
state = {
    'reference_pose': None,
    'use_hands': False,
    'current_score': 0.0,
    'is_matched': False,
    'ref_img_display_url': None,
    'capture_requested': False
}

def init_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(options)

landmarker = init_landmarker()

def load_reference():
    if os.path.exists(REFERENCE_IMAGE_PATH):
        ref_img = cv2.imread(REFERENCE_IMAGE_PATH)
        if ref_img is not None:
            # save for web serving
            cv2.imwrite('static/ref_display.jpg', ref_img)
            state['ref_img_display_url'] = '/static/ref_display.jpg'
            
            ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            mp_ref_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ref_img_rgb)
            res = landmarker.detect(mp_ref_image)
            
            if res.pose_landmarks:
                state['use_hands'] = detect_hand_activity(res.pose_landmarks[0])
                state['reference_pose'] = normalize_keypoints(res.pose_landmarks[0], include_hands=state['use_hands'])
                print("Reference loaded successfully.")
            else:
                state['reference_pose'] = None
    else:
        state['reference_pose'] = None
        state['ref_img_display_url'] = None

load_reference()

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape

    # Standard MediaPipe Pose connections
    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
        (12,14),(14,16),(16,18),(16,20),(16,22),(11,23),(12,24),
        (23,24),(23,25),(25,27),(27,29),(27,31),(24,26),(26,28),
        (28,30),(28,32)
    ]

    for pose_landmarks in pose_landmarks_list:
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                pt1 = (int(pose_landmarks[start_idx].x * w), int(pose_landmarks[start_idx].y * h))
                pt2 = (int(pose_landmarks[end_idx].x * w), int(pose_landmarks[end_idx].y * h))
                cv2.line(annotated_image, pt1, pt2, (0, 255, 150), 3)

        for lm in pose_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_image, (cx, cy), 5, (255, 50, 50), -1)

    return annotated_image

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Handle capture request
        if state['capture_requested']:
            cv2.imwrite(REFERENCE_IMAGE_PATH, frame)
            load_reference()
            state['capture_requested'] = False
            state['is_matched'] = False
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            res = landmarker.detect(mp_image)
        except Exception as e:
            res = None
            print(f"Extraction error: {e}")
        
        annotated_image = frame
        
        if res and res.pose_landmarks:
            annotated_image = draw_landmarks_on_image(frame_rgb, res)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            curr_pose = normalize_keypoints(res.pose_landmarks[0], include_hands=state['use_hands'])
            
            if state['reference_pose'] is not None and curr_pose is not None:
                sim = calculate_similarity(state['reference_pose'], curr_pose, include_hands=state['use_hands'])
                state['current_score'] = float(sim)
                
                if sim > SIMILARITY_THRESHOLD and not state['is_matched']:
                    state['is_matched'] = True
                    cv2.imwrite('static/latest_match.jpg', frame)
            else:
                state['current_score'] = 0.0
        else:
            state['current_score'] = 0.0
            
        ret, buffer = cv2.imencode('.jpg', annotated_image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify({
        'score': state['current_score'],
        'matched': state['is_matched'],
        'has_reference': state['reference_pose'] is not None,
        'ref_url': state['ref_img_display_url'] + '?t=' + str(time.time()) if state['ref_img_display_url'] else None
    })

@app.route('/api/capture', methods=['POST'])
def capture():
    state['capture_requested'] = True
    return jsonify({'status': 'success'})

@app.route('/api/reset', methods=['POST'])
def reset():
    state['is_matched'] = False
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)
