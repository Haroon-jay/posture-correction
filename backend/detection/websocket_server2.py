from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import numpy as np
import cv2
import mediapipe as mp
import math
import pandas as pd
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import joblib
import traceback
import base64
from configs.pushup_config import pushup_config
from configs.curl_config import curl_config
from configs.squat_config import squat_config

# from plank import PlankDetection
from bicep_curl import BicepCurlDetection

from squat import SquatDetection

# from lunge import LungeDetection

from process_pose import ProcessFrame


print(f"Push-up Config: {pushup_config.name}")
print(f"Bicep Curl Config: {curl_config.name}")
print(f"Squat Config: {squat_config.name}")
processes = {
    "push-up": ProcessFrame(config=pushup_config),
    "bicep-curl": ProcessFrame(config=curl_config),
    "squat": ProcessFrame(config=squat_config),
}

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
SEQUENCE_LENGTH = 40
LOOKBACK = 10

app = FastAPI()

SEQUENCE_LENGTH = 40  # Number of frames needed for prediction

labels_map = ["bicep-curl", "push-up", "squat"]
scaler_path = "model/40_5_scaler.joblib"
model_path = "model/model_seq40_lookback5.keras"

scaler = joblib.load(scaler_path)
model = load_model(model_path)


def extract_features_for_exercises(landmarks):
    """
    Extract angles, distances, and symmetry features for bicep curls, push-ups, and squats.
    """

    def get_coords(part):
        """Helper to get (x, y, z) coordinates of a body part."""
        return [part["x"], part["y"], part["z"]]

    def calculate_normalized_distance(p1, p2, body_height):
        ## skip normalization for now
        return calculate_distance(p1, p2)
        # / body_height if body_height != 0 else 0

    ## Key Landmarks
    key_parts = {
        "left_shoulder": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        "left_elbow": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        "left_wrist": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
        "left_hip": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        "left_knee": landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        "left_ankle": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
        "right_shoulder": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        "right_elbow": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        "right_wrist": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
        "right_hip": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        "right_knee": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        "right_ankle": landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    }
    # Extract coordinates
    coords = {k: get_coords(v) for k, v in key_parts.items()}

    # ## Body Height for Normalization
    # body_height = calculate_distance(
    #     landmarks[mp_pose.PoseLandmark.NOSE.value],
    #     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
    # )
    left_elbow_angle = calculate_angle(
        coords["left_shoulder"], coords["left_elbow"], coords["left_wrist"]
    )
    right_elbow_angle = calculate_angle(
        coords["right_shoulder"], coords["right_elbow"], coords["right_wrist"]
    )
    left_wrist_shoulder_dist = calculate_normalized_distance(
        coords["left_wrist"], coords["left_shoulder"], 4
    )
    right_wrist_shoulder_dist = calculate_normalized_distance(
        coords["right_wrist"], coords["right_shoulder"], 4
    )
    elbow_symmetry = abs(left_elbow_angle - right_elbow_angle)

    left_body_tilt_angle = calculate_angle(
        [coords["left_shoulder"][0], coords["left_shoulder"][1], 0],
        [coords["left_hip"][0], coords["left_hip"][1], 0],
        [0, 1, 0],
    )
    right_body_tilt_angle = calculate_angle(
        [coords["right_shoulder"][0], coords["right_shoulder"][1], 0],
        [coords["right_hip"][0], coords["right_hip"][1], 0],
        [0, 1, 0],
    )

    ## Squats Features
    left_knee_angle = calculate_angle(
        coords["left_hip"], coords["left_knee"], coords["left_ankle"]
    )
    right_knee_angle = calculate_angle(
        coords["right_hip"], coords["right_knee"], coords["right_ankle"]
    )
    left_hip_knee_dist = calculate_normalized_distance(
        coords["left_hip"], coords["left_knee"], 5
    )
    right_hip_knee_dist = calculate_normalized_distance(
        coords["right_hip"], coords["right_knee"], 5
    )
    left_shoulder_knee_dist = calculate_normalized_distance(
        coords["left_shoulder"], coords["left_knee"], 5
    )
    right_shoulder_knee_dist = calculate_normalized_distance(
        coords["right_shoulder"], coords["right_knee"], 5
    )
    left_hip_knee_vertical = abs(coords["left_hip"][1] - coords["left_knee"][1])
    right_hip_knee_vertical = abs(coords["right_hip"][1] - coords["right_knee"][1])

    def calculate_center_of_mass_projection(coords):
        hip_center_x = (coords["left_hip"][0] + coords["right_hip"][0]) / 2
        hip_center_y = (coords["left_hip"][1] + coords["right_hip"][1]) / 2
        knee_center_x = (coords["left_knee"][0] + coords["right_knee"][0]) / 2
        knee_center_y = (coords["left_knee"][1] + coords["right_knee"][1]) / 2

        return abs(hip_center_x - knee_center_x), abs(hip_center_y - knee_center_y)

    com_x_projection, com_y_projection = calculate_center_of_mass_projection(coords)

    left_wrist_height = coords["left_shoulder"][1] - coords["left_wrist"][1]
    right_wrist_height = coords["right_shoulder"][1] - coords["right_wrist"][1]
    knee_symmetry = abs(left_knee_angle - right_knee_angle)
    features = [
        # Bicep Curls
        left_elbow_angle,
        right_elbow_angle,
        left_wrist_shoulder_dist,
        right_wrist_shoulder_dist,
        elbow_symmetry,
        # Push-Ups
        left_body_tilt_angle,
        right_body_tilt_angle,
        # Squats
        left_knee_angle,
        right_knee_angle,
        left_hip_knee_dist,
        right_hip_knee_dist,
        left_shoulder_knee_dist,
        right_shoulder_knee_dist,
        left_hip_knee_vertical,
        right_hip_knee_vertical,
        com_x_projection,
        com_y_projection,
        left_wrist_height,
        right_wrist_height,
        knee_symmetry,
    ]

    return features


feature_names = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_wrist_shoulder_dist",
    "right_wrist_shoulder_dist",
    "elbow_symmetry",
    "left_body_tilt_angle",
    "right_body_tilt_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_knee_dist",
    "right_hip_knee_dist",
    "left_shoulder_knee_dist",
    "right_shoulder_knee_dist",
    "left_hip_knee_vertical",
    "right_hip_knee_vertical",
    "com_x_projection",
    "com_y_projection",
    "left_wrist_height",
    "right_wrist_height",
    "knee_symmetry",
]


def calculate_distance(point1, point2):
    """
    Calculate 3D Euclidean distance between two points
    Provides more accurate spatial representation across different view angles
    """
    return math.sqrt(
        (point2[0] - point1[0]) ** 2  # Horizontal displacement
        + (point2[1] - point1[1]) ** 2  # Vertical displacement
        + (point2[2] - point1[2]) ** 2  # Depth displacement
    )


def calculate_angle(a, b, c):
    """
    Calculate 3D angle between three points
    Uses vector algebra for more robust angle calculation
    """
    # Convert points to numpy arrays
    a = np.array([a[0], a[1], a[2]])
    b = np.array([b[0], b[1], b[2]])
    c = np.array([c[0], c[1], c[2]])
    # Create vectors
    vec1 = a - b  # Vector from point b to point a
    vec2 = c - b  # Vector from point b to point c

    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)

    # Calculate dot product
    dot_product = np.dot(vec1_norm, vec2_norm)

    # Ensure dot product is within valid range due to potential floating-point errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


# Dictionary to store frame buffers per user
user_buffers: Dict[str, List[dict]] = {}
# EXERCISE_DETECTIONS = {
#     # "plank": PlankDetection(),
#     "bicep_curl": BicepCurlDetection(),
#     "squat": SquatDetection(),
#     # "lunge": LungeDetection(),
# }
# exercise_detection = EXERCISE_DETECTIONS.get("squat")
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Use the client's WebSocket ID as a unique identifier
    client_id = id(websocket)
    user_buffers[client_id] = []  # Create a separate buffer for this user
    print(f"Client connected: {client_id}")
    count = 0
    try:
        await websocket.accept()
        while True:
            # Wait for a message (features for one frame) from the client
            data = await websocket.receive_json()
            if not data:
                return
            print(f"Received frame from {client_id}")
            type = data.get("type")
            if type == "landmarks":
                features = extract_features_for_exercises(data["landmarks"])
                user_buffers[client_id].append(features)
                if len(user_buffers[client_id]) == SEQUENCE_LENGTH:
                    # Prepare the data for prediction
                    input_sequence = np.array(user_buffers[client_id])
                    prediction = make_prediction(input_sequence)
                    print(f"Prediction for {client_id}: {prediction}")

                    # Send the prediction back to the client
                    await websocket.send_json({"prediction": prediction})

                    # Clear the buffer for the next sequence
                    user_buffers[client_id] = []
            if type == "img":
                img = data["image"]
                prediction = data["prediction"]
                if prediction not in processes:
                    print(f"Invalid prediction: {prediction}")
                    continue
                process = processes[prediction]
                print("Process Config: ", process.config.name)
                print(f"Prediction received for {client_id}: {prediction}")
                ## write to file, img is base64 encoded
                if img.startswith("data:image"):
                    img = img.split(",", 1)[1]
                # with open("test.jpg", "wb") as f:
                #     f.write(base64.b64decode(img))
                img_len = len(img)

                frame = base64_to_cv2(img)
                print(f"Frame shape: {frame.shape}")
                (result, frame) = process_frame(frame, pose)
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                state, message = None, None
                if result.pose_landmarks:
                    (state, message) = process.process(
                        result.pose_landmarks.landmark, frame, frame_width, frame_height
                    )
                    # print(f"State: {state}")
                    print(f"Message: {message}")
                    # exercise_detection.detect(
                    #     mp_results=result, image=frame, timestamp=count
                    # )

                processed_base64 = cv2_to_base64(frame)

                count += 1

                # process.process(data["landmarks"])
                print(f"State for {client_id}: {state}")
                await websocket.send_json(
                    {
                        "status": "frame received",
                        "state": state.to_dict() if state else None,
                        "message": message,
                        "image": processed_base64,
                    }
                )

    except WebSocketDisconnect:
        print(f"Client disconnected: {client_id}")
        # Remove the user's buffer when they disconnect
        if client_id in user_buffers:
            del user_buffers[client_id]
    except Exception as e:
        print(f"Error for {client_id}: {e}")
        print(traceback.format_exc())
    finally:
        # Clean up on disconnection or error
        if client_id in user_buffers:
            del user_buffers[client_id]


def normalize_sequences(sequences):
    arr_sequences = np.array(sequences)
    num_samples, sequence_length, num_features = arr_sequences.shape
    arr_reshaped = arr_sequences.reshape(-1, num_features)
    # model = load_model("model.keras")
    # Fit and transform
    X_normalized = scaler.fit_transform(arr_reshaped)
    X_normalized = X_normalized.reshape(num_samples, sequence_length, num_features)
    return X_normalized


def make_prediction(frame_buffer: np.ndarray):
    """
    Placeholder for your prediction logic.
    Replace this with your actual model inference.
    """
    input_sequence = np.array(frame_buffer).reshape(1, SEQUENCE_LENGTH, -1)
    normalized_sequence = normalize_sequences(input_sequence)
    predictions = model.predict(normalized_sequence)
    pred = labels_map[predictions[0].argmax()]
    return {"label": pred, "confidence": 0.99}


def base64_to_cv2(base64_string):
    """Convert base64 string to OpenCV image"""
    img_bytes = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def cv2_to_base64(image):
    """Convert OpenCV image to base64"""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def process_frame(frame, pose):
    """Process a frame using Mediapipe Pose detection"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return (results, image)
