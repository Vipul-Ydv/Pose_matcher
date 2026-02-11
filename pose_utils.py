
import numpy as np

# MediaPipe Pose landmark indices
# Body: 11-16 (shoulders, elbows, wrists), 23-28 (hips, knees, ankles)
# Hands: 17(L_pinky), 18(R_pinky), 19(L_index), 20(R_index), 21(L_thumb), 22(R_thumb)

# Body-only indices (12 joints)
BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Body + hand indices (18 joints)
BODY_HAND_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]


def detect_hand_activity(landmarks):
    """
    Detects whether hands are doing something significant in the pose.
    Returns True if hands are "active" (e.g., raised, spread, or close together).
    
    Checks:
    1. Are hands raised above the elbows? (waving, raised hand)
    2. Are hands close together? (namaste, clapping)
    3. Are finger landmarks spread from the wrist? (open hand gesture)
    """
    if not landmarks or len(landmarks) < 33:
        return False
    
    # Get relevant landmarks
    l_shoulder = np.array([landmarks[11].x, landmarks[11].y])
    r_shoulder = np.array([landmarks[12].x, landmarks[12].y])
    l_elbow = np.array([landmarks[13].x, landmarks[13].y])
    r_elbow = np.array([landmarks[14].x, landmarks[14].y])
    l_wrist = np.array([landmarks[15].x, landmarks[15].y])
    r_wrist = np.array([landmarks[16].x, landmarks[16].y])
    l_pinky = np.array([landmarks[17].x, landmarks[17].y])
    r_pinky = np.array([landmarks[18].x, landmarks[18].y])
    l_index = np.array([landmarks[19].x, landmarks[19].y])
    r_index = np.array([landmarks[20].x, landmarks[20].y])
    l_thumb = np.array([landmarks[21].x, landmarks[21].y])
    r_thumb = np.array([landmarks[22].x, landmarks[22].y])
    
    shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
    if shoulder_width < 1e-6:
        return False
    
    # Check 1: Are hands close together? (namaste, prayer, clapping)
    # Distance between wrists relative to shoulder width
    wrist_dist = np.linalg.norm(l_wrist - r_wrist) / shoulder_width
    if wrist_dist < 0.4:  # Wrists are very close together
        return True
    
    # Check 2: Are wrists raised above shoulders? (y decreases upward in image coords)
    if l_wrist[1] < l_shoulder[1] or r_wrist[1] < r_shoulder[1]:
        # At least one hand is raised above shoulder
        # Check if finger landmarks are spread (indicating intentional hand pose)
        l_hand_spread = (np.linalg.norm(l_pinky - l_wrist) + 
                         np.linalg.norm(l_index - l_wrist) + 
                         np.linalg.norm(l_thumb - l_wrist)) / (3 * shoulder_width)
        r_hand_spread = (np.linalg.norm(r_pinky - r_wrist) + 
                         np.linalg.norm(r_index - r_wrist) + 
                         np.linalg.norm(r_thumb - r_wrist)) / (3 * shoulder_width)
        
        if l_hand_spread > 0.15 or r_hand_spread > 0.15:
            return True
    
    # Check 3: Are finger landmarks forming a distinct pattern?
    # Compare angle between index-wrist-thumb (indicates specific hand shape)
    for wrist, index, thumb in [(l_wrist, l_index, l_thumb), (r_wrist, r_index, r_thumb)]:
        angle = _joint_angle(index, wrist, thumb)
        # If fingers are significantly spread (> 30 degrees), hands are active
        if angle > np.radians(30):
            # Also check this hand is at least at elbow height
            if wrist[1] < (l_elbow[1] + r_elbow[1]) / 2 + 0.05:
                return True
    
    return False


def normalize_keypoints(landmarks, include_hands=False):
    """
    Normalizes pose landmarks to be translation and scale invariant.
    Centers the pose at the hip midpoint and scales by shoulder width.
    
    Args:
        landmarks: A list of MediaPipe pose landmarks.
        include_hands: If True, includes hand landmarks (pinky, index, thumb).
        
    Returns:
        A numpy array of normalized (x, y) coordinates for key joints.
    """
    if not landmarks:
        return None

    indices = BODY_HAND_INDICES if include_hands else BODY_INDICES
    
    points = []
    for i in indices:
        lm = landmarks[i]
        points.append([lm.x, lm.y])
    
    points = np.array(points)
    
    if include_hands:
        # Hip indices in the 18-joint set: L_hip=12, R_hip=13
        hip_center = (points[12] + points[13]) / 2
        points_centered = points - hip_center
        # Shoulder indices: L_shoulder=0, R_shoulder=1
        left_shoulder = points_centered[0]
        right_shoulder = points_centered[1]
    else:
        # Hip indices in the 12-joint set: L_hip=6, R_hip=7
        hip_center = (points[6] + points[7]) / 2
        points_centered = points - hip_center
        left_shoulder = points_centered[0]
        right_shoulder = points_centered[1]
    
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    
    if shoulder_width < 1e-6:
        return points_centered
        
    points_normalized = points_centered / shoulder_width
    
    return points_normalized


def calculate_similarity(pose1, pose2, include_hands=False):
    """
    Calculates similarity between two normalized poses.
    Adaptively includes hand landmarks if include_hands is True.
    
    Args:
        pose1: Normalized numpy array of shape (N, 2).
        pose2: Normalized numpy array of shape (N, 2).
        include_hands: Whether hand landmarks are included.
        
    Returns:
        A similarity score between 0.0 and 1.0.
    """
    if pose1 is None or pose2 is None:
        return 0.0
        
    if pose1.shape != pose2.shape:
        return 0.0
    
    if include_hands:
        # 18-joint layout:
        # 0:L_shoulder, 1:R_shoulder, 2:L_elbow, 3:R_elbow,
        # 4:L_wrist, 5:R_wrist,
        # 6:L_pinky, 7:R_pinky, 8:L_index, 9:R_index, 10:L_thumb, 11:R_thumb,
        # 12:L_hip, 13:R_hip, 14:L_knee, 15:R_knee, 16:L_ankle, 17:R_ankle
        
        # Limb + hand indices (exclude shoulders 0,1 and hips 12,13)
        limb_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
        limb_weights = np.array([
            2.0, 2.0,  # elbows
            2.5, 2.5,  # wrists
            3.0, 3.0,  # pinky
            3.0, 3.0,  # index
            3.0, 3.0,  # thumb
            1.5, 1.5,  # knees
            1.0, 1.0,  # ankles
        ])
        limb_weights = limb_weights / limb_weights.sum()
        
        angle_triplets = [
            (0, 2, 4),    # L_shoulder -> L_elbow -> L_wrist
            (1, 3, 5),    # R_shoulder -> R_elbow -> R_wrist
            (12, 14, 16), # L_hip -> L_knee -> L_ankle
            (13, 15, 17), # R_hip -> R_knee -> R_ankle
            (2, 0, 12),   # L_elbow -> L_shoulder -> L_hip
            (3, 1, 13),   # R_elbow -> R_shoulder -> R_hip
            (0, 12, 14),  # L_shoulder -> L_hip -> L_knee
            (1, 13, 15),  # R_shoulder -> R_hip -> R_knee
            # Hand angles
            (8, 4, 10),   # L_index -> L_wrist -> L_thumb
            (9, 5, 11),   # R_index -> R_wrist -> R_thumb
            (6, 4, 8),    # L_pinky -> L_wrist -> L_index
            (7, 5, 9),    # R_pinky -> R_wrist -> R_index
        ]
    else:
        # 12-joint layout (original):
        # 0:L_shoulder, 1:R_shoulder, 2:L_elbow, 3:R_elbow,
        # 4:L_wrist, 5:R_wrist, 6:L_hip, 7:R_hip,
        # 8:L_knee, 9:R_knee, 10:L_ankle, 11:R_ankle
        
        limb_indices = [2, 3, 4, 5, 8, 9, 10, 11]
        limb_weights = np.array([2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.5, 1.5])
        limb_weights = limb_weights / limb_weights.sum()
        
        angle_triplets = [
            (0, 2, 4),   # L_shoulder -> L_elbow -> L_wrist
            (1, 3, 5),   # R_shoulder -> R_elbow -> R_wrist
            (6, 8, 10),  # L_hip -> L_knee -> L_ankle
            (7, 9, 11),  # R_hip -> R_knee -> R_ankle
            (2, 0, 6),   # L_elbow -> L_shoulder -> L_hip
            (3, 1, 7),   # R_elbow -> R_shoulder -> R_hip
            (0, 6, 8),   # L_shoulder -> L_hip -> L_knee
            (1, 7, 9),   # R_shoulder -> R_hip -> R_knee
        ]
    
    # --- Part 1: Weighted limb distance ---
    limb_dists = np.linalg.norm(pose1[limb_indices] - pose2[limb_indices], axis=1)
    weighted_dist = np.sum(limb_dists * limb_weights)
    
    sigma = 0.15
    dist_score = np.exp(-(weighted_dist ** 2) / (2 * sigma ** 2))
    
    # --- Part 2: Joint angle comparison ---
    angle_diffs = []
    for a, b, c in angle_triplets:
        angle1 = _joint_angle(pose1[a], pose1[b], pose1[c])
        angle2 = _joint_angle(pose2[a], pose2[b], pose2[c])
        angle_diffs.append(abs(angle1 - angle2))
    
    mean_angle_diff = np.mean(angle_diffs)
    angle_score = max(0.0, 1.0 - mean_angle_diff / (np.pi / 2))
    
    # --- Combine: 40% distance, 60% angles ---
    final_score = 0.4 * dist_score + 0.6 * angle_score
    
    return max(0.0, min(1.0, float(final_score)))


def _joint_angle(a, b, c):
    """Calculate the angle at point b formed by segments a-b and b-c."""
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-8:
        return 0.0
    cos_angle = np.dot(v1, v2) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))
