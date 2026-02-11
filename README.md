# Real-Time Pose Matching Application

This application strictly compares your live pose ensuring you match a reference pose. It uses MediaPipe for pose detection and Euclidean distance on normalized keypoints for comparison.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Add Reference Image**:
    -   Place a `reference_pose.jpg` file in this directory.
    -   *Tip*: You can run the app without one, and press **'c'** to capture your current pose as the reference!

3.  **Run the Application**:
    ```bash
    python main.py
    ```

## Controls

-   **'q'**: Quit the application.
-   **'c'**: Capture the current live frame as the new reference pose.

## How It Works

-   **Extraction**: Detects 33 body landmarks (we use 12 main joints).
-   **Normalization**: Re-centers the pose at the hips and scales by shoulder width to allow matching at different distances from the camera.
-   **Comparison**: Calculates similarity score (0.0 to 1.0).
-   **Feedback**:
    -   **Visual**: Displays "PASSED!" and similarity score.
    -   **Audio**: Plays `match_sound.wav` (if present) or system beep.
    -   **Capture**: Automatically saves the matching frame as `match_<timestamp>.jpg` (with cooldown to prevent spam).

## Troubleshooting

-   **No Module Named ...**: Ensure you installed requirements.
-   **Camera Not Opening**: Check if another app is using the webcam.
-   **Low Score**: Ensure good lighting and visibility of your full body.
