const { Pose, POSE_CONNECTIONS, Camera, drawConnectors, drawLandmarks } = window;
import { detectHandActivity, normalizeKeypoints, calculateSimilarity } from './poseUtils';
import './style.css';

const videoElement = document.getElementById('webcam-video');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const scoreText = document.getElementById('score-text');
const progressFill = document.getElementById('progress-fill');
const appStatus = document.getElementById('app-status');
const captureBtn = document.getElementById('capture-btn');
const matchCelebration = document.getElementById('match-celebration');
const continueBtn = document.getElementById('continue-btn');
const referenceContainer = document.getElementById('reference-container');

// State
let referencePose = null;
let useHands = false;
let isMatched = false;
let isCapturing = false;
let matchCooldown = false;

// MediaPipe Setup
const pose = new Pose({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
}});
pose.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

pose.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await pose.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();

appStatus.innerText = "Camera Active... Strike a pose and Capture!";

// Process Frames
function onResults(results) {
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
  
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Mirror effect
  canvasCtx.translate(canvasElement.width, 0);
  canvasCtx.scale(-1, 1);
  
  // Draw Webcam Image
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.poseLandmarks) {
    const lms = results.poseLandmarks;
    
    // Draw Skeleton
    drawConnectors(canvasCtx, lms, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
    drawLandmarks(canvasCtx, lms, {color: '#FF0000', lineWidth: 2, radius: 4});

    // Capture Logic
    if (isCapturing) {
      isCapturing = false;
      useHands = detectHandActivity(lms);
      referencePose = normalizeKeypoints(lms, useHands);
      
      // Save canvas to image and show it
      const dataUrl = getCleanImage(results.image, canvasElement.width, canvasElement.height);
      document.getElementById('no-ref-placeholder').style.display = 'none';
      referenceContainer.innerHTML = `<img src="${dataUrl}" class="ref-snapshot" />`;
      
      appStatus.innerText = "Tracking! Match the target pose.";
      appStatus.style.color = "var(--accent)";
      appStatus.style.borderColor = "var(--accent)";
      isMatched = false;
      matchCooldown = true;
      setTimeout(() => { matchCooldown = false; }, 2000);
    }

    // Match Logic
    if (referencePose && !isMatched) {
      const currentPose = normalizeKeypoints(lms, useHands);
      if (currentPose) {
        let score = calculateSimilarity(referencePose, currentPose, useHands);
        updateScoreUI(score);

        if (score > 0.85 && !matchCooldown) {
          isMatched = true;
          matchCelebration.classList.add('active');
          const dataUrl = getCleanImage(results.image, canvasElement.width, canvasElement.height);

          const downloadBtn = document.getElementById('download-match-btn');
          if (downloadBtn) {
              downloadBtn.href = dataUrl;
              downloadBtn.style.display = 'flex';
          }

          document.getElementById('history-container').innerHTML = 
            `<div class="history-item slideUp">
              <img src="${dataUrl}" alt="Matched Pose">
              <div class="label">SUCCESS</div>
              <a href="${dataUrl}" download="pose-match.png" style="position: absolute; top: 5px; right: 5px; background: rgba(0,0,0,0.8); color: white; padding: 0.3rem 0.6rem; border-radius: 4px; text-decoration: none; font-size: 0.8rem; z-index: 10;">⬇️ Save</a>
            </div>`;
        }
      }
    } else if (!referencePose) {
        updateScoreUI(0.0);
    }
  }
  canvasCtx.restore();
}

function updateScoreUI(score) {
    const percentage = Math.min(Math.round(score * 100), 100);
    scoreText.textContent = `${percentage}%`;
    progressFill.style.width = `${percentage}%`;

    if (percentage > 90) {
        progressFill.style.filter = "drop-shadow(0 0 10px var(--accent))";
        scoreText.style.color = "var(--accent)";
    } else if (percentage > 70) {
        progressFill.style.filter = "drop-shadow(0 0 5px #fbbf24)";
        scoreText.style.color = "#fbbf24";
    } else {
        progressFill.style.filter = "none";
        scoreText.style.color = "white";
    }
}

captureBtn.addEventListener('click', () => {
    isCapturing = true;
});

continueBtn.addEventListener('click', () => {
    isMatched = false;
    matchCelebration.classList.remove('active');
    matchCooldown = true;
    setTimeout(() => { matchCooldown = false; }, 2000);
});

function getCleanImage(imageSource, width, height) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.translate(width, 0);
    tempCtx.scale(-1, 1);
    tempCtx.drawImage(imageSource, 0, 0, width, height);
    return tempCanvas.toDataURL('image/png');
}
