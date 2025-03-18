import cv2
import numpy as np
import mediapipe as mp
from squat import SquatDetection  # Adjust if you want a different exercise
from process_pose import ProcessFrame
from configs.pushup_config import pushup_config
from configs.curl_config import curl_config
from configs.squat_config import squat_config

import custom_print

# Initialize components
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
exercise_detection = SquatDetection()  # Change to BicepCurlDetection if needed

process = ProcessFrame(config=pushup_config)


def process_frame(frame, pose):
    """Process a frame using Mediapipe Pose detection"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return (results, image)


# Video capture
cap = cv2.VideoCapture(0)  # 0 = default webcam

# Set video writer to save processed output


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("processed_vid.avi", fourcc, 20.0, size)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally (optional)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    # Process frame using your existing process_frame logic
    result, processed_frame = process_frame(frame, pose)
    print(result)
    if result.pose_landmarks:
        (state, message) = process.process(
            result.pose_landmarks.landmark, processed_frame, frame_width, frame_height
        )
        print(f"State: {state}")
        print(f"Message: {message}")
        # exercise_detection.detect(
        #     mp_results=result, image=processed_frame, timestamp=frame_count
        # )

    # Write processed frame to video
    # out.write(processed_frame)

    # Optionally display the frame (for debugging or live feedback)
    cv2.imshow("Processed Frame", processed_frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
