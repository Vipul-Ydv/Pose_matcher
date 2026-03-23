// poseUtils.js
// Equivalent to pose_utils.py logic

export const BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
export const BODY_HAND_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28];

const vecSub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const vecAdd = (a, b) => [a[0] + b[0], a[1] + b[1]];
const vecNorm = (v) => Math.sqrt(v[0] * v[0] + v[1] * v[1]);
const vecDot = (a, b) => a[0] * b[0] + a[1] * b[1];

export function jointAngle(a, b, c) {
    const v1 = vecSub(a, b);
    const v2 = vecSub(c, b);
    const denom = vecNorm(v1) * vecNorm(v2);
    if (denom < 1e-8) return 0.0;
    let cosAngle = vecDot(v1, v2) / denom;
    cosAngle = Math.max(-1.0, Math.min(1.0, cosAngle));
    return Math.acos(cosAngle);
}

export function detectHandActivity(landmarks) {
    if (!landmarks || landmarks.length < 33) return false;

    const lmInfo = (i) => [landmarks[i].x, landmarks[i].y];

    const l_shoulder = lmInfo(11);
    const r_shoulder = lmInfo(12);
    const l_elbow = lmInfo(13);
    const r_elbow = lmInfo(14);
    const l_wrist = lmInfo(15);
    const r_wrist = lmInfo(16);
    const l_pinky = lmInfo(17);
    const r_pinky = lmInfo(18);
    const l_index = lmInfo(19);
    const r_index = lmInfo(20);
    const l_thumb = lmInfo(21);
    const r_thumb = lmInfo(22);

    const shoulder_width = vecNorm(vecSub(l_shoulder, r_shoulder));
    if (shoulder_width < 1e-6) return false;

    // Check 1: Wrists close together
    const wrist_dist = vecNorm(vecSub(l_wrist, r_wrist)) / shoulder_width;
    if (wrist_dist < 0.4) return true;

    // Check 2: Hands raised above shoulders
    if (l_wrist[1] < l_shoulder[1] || r_wrist[1] < r_shoulder[1]) {
        const l_hand_spread = (vecNorm(vecSub(l_pinky, l_wrist)) + 
                               vecNorm(vecSub(l_index, l_wrist)) + 
                               vecNorm(vecSub(l_thumb, l_wrist))) / (3 * shoulder_width);
        const r_hand_spread = (vecNorm(vecSub(r_pinky, r_wrist)) + 
                               vecNorm(vecSub(r_index, r_wrist)) + 
                               vecNorm(vecSub(r_thumb, r_wrist))) / (3 * shoulder_width);
        
        if (l_hand_spread > 0.15 || r_hand_spread > 0.15) return true;
    }

    // Check 3: Active finger splay
    const pairs = [
        { wrist: l_wrist, index: l_index, thumb: l_thumb, elbow: l_elbow },
        { wrist: r_wrist, index: r_index, thumb: r_thumb, elbow: r_elbow }
    ];

    for (const p of pairs) {
        let angle = jointAngle(p.index, p.wrist, p.thumb);
        if (angle > (30 * Math.PI / 180)) {
            if (p.wrist[1] < (l_elbow[1] + r_elbow[1]) / 2 + 0.05) {
                return true;
            }
        }
    }
    return false;
}

export function normalizeKeypoints(landmarks, includeHands = false) {
    if (!landmarks) return null;

    const indices = includeHands ? BODY_HAND_INDICES : BODY_INDICES;
    const points = [];

    for (let i of indices) {
        points.push([landmarks[i].x, landmarks[i].y]);
    }

    let left_shoulder, right_shoulder, hip_center;

    if (includeHands) {
        hip_center = [(points[12][0] + points[13][0]) / 2, (points[12][1] + points[13][1]) / 2];
        const pointsCentered = points.map(p => vecSub(p, hip_center));
        left_shoulder = pointsCentered[0];
        right_shoulder = pointsCentered[1];
        
        const shoulder_width = vecNorm(vecSub(left_shoulder, right_shoulder));
        if (shoulder_width < 1e-6) return pointsCentered;

        return pointsCentered.map(p => [p[0] / shoulder_width, p[1] / shoulder_width]);
    } else {
        hip_center = [(points[6][0] + points[7][0]) / 2, (points[6][1] + points[7][1]) / 2];
        const pointsCentered = points.map(p => vecSub(p, hip_center));
        left_shoulder = pointsCentered[0];
        right_shoulder = pointsCentered[1];
        
        const shoulder_width = vecNorm(vecSub(left_shoulder, right_shoulder));
        if (shoulder_width < 1e-6) return pointsCentered;

        return pointsCentered.map(p => [p[0] / shoulder_width, p[1] / shoulder_width]);
    }
}

export function calculateSimilarity(pose1, pose2, includeHands = false) {
    if (!pose1 || !pose2) return 0.0;
    if (pose1.length !== pose2.length) return 0.0;

    let limb_indices, limb_weights, angle_triplets;

    if (includeHands) {
        limb_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17];
        limb_weights = [2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.5, 1.5, 1.0, 1.0];
        angle_triplets = [
            [0, 2, 4], [1, 3, 5], [12, 14, 16], [13, 15, 17],
            [2, 0, 12], [3, 1, 13], [0, 12, 14], [1, 13, 15],
            [8, 4, 10], [9, 5, 11], [6, 4, 8], [7, 5, 9]
        ];
    } else {
        limb_indices = [2, 3, 4, 5, 8, 9, 10, 11];
        limb_weights = [2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.5, 1.5];
        angle_triplets = [
            [0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11],
            [2, 0, 6], [3, 1, 7], [0, 6, 8], [1, 7, 9]
        ];
    }

    const sumW = limb_weights.reduce((a, b) => a + b, 0);
    limb_weights = limb_weights.map(w => w / sumW);

    let weighted_dist = 0;
    for (let i = 0; i < limb_indices.length; i++) {
        const idx = limb_indices[i];
        const dist = vecNorm(vecSub(pose1[idx], pose2[idx]));
        weighted_dist += dist * limb_weights[i];
    }

    const sigma = 0.15;
    const dist_score = Math.exp(-(weighted_dist * weighted_dist) / (2 * sigma * sigma));

    let sum_angle_diff = 0;
    for (const [a, b, c] of angle_triplets) {
        const angle1 = jointAngle(pose1[a], pose1[b], pose1[c]);
        const angle2 = jointAngle(pose2[a], pose2[b], pose2[c]);
        sum_angle_diff += Math.abs(angle1 - angle2);
    }

    const mean_angle_diff = sum_angle_diff / angle_triplets.length;
    let angle_score = 1.0 - mean_angle_diff / (Math.PI / 2);
    if (angle_score < 0) angle_score = 0;

    let final_score = 0.4 * dist_score + 0.6 * angle_score;
    return Math.max(0.0, Math.min(1.0, final_score));
}
