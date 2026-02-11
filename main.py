import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pygame
import time
import os
import sys

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pose_utils import normalize_keypoints, calculate_similarity, detect_hand_activity

# Constants
REFERENCE_IMAGE_PATH = 'reference_pose.png'
MATCH_SOUND_PATH = 'match_sound.wav'
SIMILARITY_THRESHOLD = 0.92
COOLDOWN_SECONDS = 3.0
MODEL_PATH = 'pose_landmarker_lite.task'

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape

    # Standard MediaPipe Pose connections (pairs of landmark indices)
    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
        (12,14),(14,16),(16,18),(16,20),(16,22),(11,23),(12,24),
        (23,24),(23,25),(25,27),(27,29),(27,31),(24,26),(26,28),
        (28,30),(28,32)
    ]

    for pose_landmarks in pose_landmarks_list:
        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                pt1 = (int(pose_landmarks[start_idx].x * w), int(pose_landmarks[start_idx].y * h))
                pt2 = (int(pose_landmarks[end_idx].x * w), int(pose_landmarks[end_idx].y * h))
                cv2.line(annotated_image, pt1, pt2, (0, 255, 0), 2)

        # Draw landmarks
        for lm in pose_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_image, (cx, cy), 4, (0, 0, 255), -1)

    return annotated_image


def main():
    # Initialize Pygame Mixer
    pygame.mixer.init()
    match_sound = None
    if os.path.exists(MATCH_SOUND_PATH):
        try:
            match_sound = pygame.mixer.Sound(MATCH_SOUND_PATH)
        except Exception as e:
            print(f"Warning: Could not load sound file: {e}")
    else:
        print(f"Warning: Sound file '{MATCH_SOUND_PATH}' not found. Audio feedback disabled.")

    # Initialize Pose Landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    
    try:
        landmarker = vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error initializing PoseLandmarker: {e}")
        print(f"Ensure '{MODEL_PATH}' is in the current directory.")
        return

    # Load Reference Image
    reference_pose = None
    ref_img_display = None  # Keep original ref image for side-by-side display
    use_hands = False  # Auto-detected based on reference pose
    if os.path.exists(REFERENCE_IMAGE_PATH):
        ref_img = cv2.imread(REFERENCE_IMAGE_PATH)
        if ref_img is not None:
            print("Processing reference image...")
            ref_img_display = ref_img.copy()  # Clean copy for side-by-side
            # Convert to MP Image
            ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            mp_ref_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ref_img_rgb)
            
            detection_result = landmarker.detect(mp_ref_image)
            
            if detection_result.pose_landmarks:
                # Use the first detected pose
                ref_landmarks = detection_result.pose_landmarks[0]
                # Auto-detect if hands are important in this pose
                use_hands = detect_hand_activity(ref_landmarks)
                reference_pose = normalize_keypoints(ref_landmarks, include_hands=use_hands)
                if use_hands:
                    print("Hand gesture detected — using hand + body matching.")
                else:
                    print("Body-only pose — using body matching.")
                print("Reference pose loaded and normalized.")
            else:
                print("Error: No pose detected in reference image.")
        else:
            print("Error: Could not read reference image.")
    else:
        print(f"Warning: '{REFERENCE_IMAGE_PATH}' not found. Please place a reference image in the folder.")
        print("Tip: You can press 'c' to capture the current frame as reference.")

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    matched = False
    
    print("\nStarting Pose Matcher...")
    print("Press 'q' to quit.")
    print("Press 'c' to capture current frame as new reference.")

    while cap.isOpened() and not matched:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to MP Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect
        detection_result = landmarker.detect(mp_image)
        
        current_pose_norm = None
        similarity = 0.0
        
        annotated_image = frame
        
        if detection_result.pose_landmarks:
            # Draw landmarks
            annotated_image = draw_landmarks_on_image(frame_rgb, detection_result)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Normalize live pose (take first person)
            current_pose_norm = normalize_keypoints(detection_result.pose_landmarks[0], include_hands=use_hands)
            
            # Compare with reference
            if reference_pose is not None and current_pose_norm is not None:
                similarity = calculate_similarity(reference_pose, current_pose_norm, include_hands=use_hands)
                
                # Display Score
                cv2.putText(annotated_image, f"Score: {similarity:.2f}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Match Logic — capture ONCE then show side-by-side
                if similarity > SIMILARITY_THRESHOLD:
                    cv2.putText(annotated_image, "MATCHED!", (10, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    # Play Sound
                    if match_sound:
                        match_sound.play()
                    else:
                        print("\a")
                        
                    # Save annotated image to file
                    filename = f"match_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated_image)
                    print(f"Matched! Score: {similarity:.2f}. Saved {filename}")
                    
                    # Use clean frame (no skeleton/text) for side-by-side
                    clean_capture = frame.copy()
                    
                    # Stop camera and show side-by-side
                    matched = True
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    # Build side-by-side comparison with clean images
                    show_side_by_side(ref_img_display, clean_capture, similarity)
                    break
        
        # User Feedback if no reference
        if reference_pose is None:
            cv2.putText(annotated_image, "No Reference Pose Set", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_image, "Press 'c' to capture", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Pose Matcher', annotated_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture current frame as reference
            if detection_result.pose_landmarks:
                live_landmarks = detection_result.pose_landmarks[0]
                use_hands = detect_hand_activity(live_landmarks)
                reference_pose = normalize_keypoints(live_landmarks, include_hands=use_hands)
                ref_img_display = frame.copy()  # Clean copy for side-by-side
                if use_hands:
                    print("Captured reference with hand matching enabled.")
                else:
                    print("Captured reference with body-only matching.")
                cv2.imwrite(REFERENCE_IMAGE_PATH, frame)
            else:
                print("Cannot capture: No pose detected in current frame.")

    if not matched:
        cap.release()
        cv2.destroyAllWindows()


def show_side_by_side(ref_img, captured_img, score):
    """Display reference and captured images side by side."""
    # Resize both to same height
    target_h = 500
    
    def resize_to_height(img, h):
        aspect = img.shape[1] / img.shape[0]
        new_w = int(h * aspect)
        return cv2.resize(img, (new_w, h))
    
    ref_resized = resize_to_height(ref_img, target_h)
    cap_resized = resize_to_height(captured_img, target_h)
    
    # Add labels
    label_h = 60
    ref_label = np.zeros((label_h, ref_resized.shape[1], 3), dtype=np.uint8)
    cap_label = np.zeros((label_h, cap_resized.shape[1], 3), dtype=np.uint8)
    
    cv2.putText(ref_label, "REFERENCE", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(cap_label, f"CAPTURED (Score: {score:.2f})", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    ref_with_label = np.vstack([ref_label, ref_resized])
    cap_with_label = np.vstack([cap_label, cap_resized])
    
    # Add a divider line between images
    divider = np.ones((ref_with_label.shape[0], 4, 3), dtype=np.uint8) * 255
    
    side_by_side = np.hstack([ref_with_label, divider, cap_with_label])
    
    cv2.imshow('Pose Match Result - Press any key to close', side_by_side)
    print("Showing side-by-side comparison. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
