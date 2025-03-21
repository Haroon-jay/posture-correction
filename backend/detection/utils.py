import mediapipe as mp
import cv2
import numpy as np
import datetime
import os
import math
from sklearn.preprocessing import MinMaxScaler

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# * Mediapipe Utils Functions
def calculate_angle(point1: list, point2: list, point3: list) -> float:
    """Calculate the angle between 3 points

    Args:
        point1 (list): Point 1 coordinate
        point2 (list): Point 2 coordinate
        point3 (list): Point 3 coordinate

    Returns:
        float: angle in degree
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate algo
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(
        point1[1] - point2[1], point1[0] - point2[0]
    )
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg


def calculate_distance(pointX: list, pointY: list) -> float:
    """Calculate distance between 2 points in a frame

    Args:
        pointX (list): First point coordinate
        pointY (list): Second point coordinate

    Returns:
        float: _description_
    """

    x1, y1 = pointX
    x2, y2 = pointY

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def extract_important_keypoints(results, important_landmarks: list) -> list:
    """Extract important landmarks' data from MediaPipe output

    Args:
        results : MediaPipe Pose output
        important_landmarks (list): list of important landmarks

    Returns:
        list: list of important landmarks' data from MediaPipe output
    """
    landmarks = results.pose_landmarks.landmark

    data = []
    for lm in important_landmarks:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])

    return np.array(data).flatten().tolist()


def get_drawing_color(error: bool) -> tuple:
    """Get drawing color for MediaPipe Pose

    Args:
        error (bool): True if correct pose, False if incorrect pose

    Returns:
        tuple: RGB colors
    """
    LIGHT_BLUE = (244, 117, 66)
    LIGHT_PINK = (245, 66, 230)

    LIGHT_RED = (29, 62, 199)
    LIGHT_YELLOW = (1, 143, 241)

    return (LIGHT_YELLOW, LIGHT_RED) if error else (LIGHT_BLUE, LIGHT_PINK)


# * OpenCV util functions
def rescale_frame(frame, percent=50):
    """Rescale a frame from OpenCV to a certain percentage compare to its original frame

    Args:
        frame: OpenCV frame
        percent (int, optional): percent to resize an old frame. Defaults to 50.

    Returns:
        _type_: OpenCV frame
    """
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def save_frame_as_image(frame, message: str = None):
    """
    Save a frame as image to display the error
    """
    now = datetime.datetime.now()

    if message:
        cv2.putText(
            frame,
            message,
            (50, 150),
            cv2.FONT_HERSHEY_COMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    print("Saving ...")
    cv2.imwrite(f"../data/logs/bicep_{now}.jpg", frame)


# * Other util functions
def get_static_file_url(file_name: str) -> str:
    """Return static url of a file

    Args:
        file_name (str)

    Returns:
        str: Full absolute path of the file. Return None if file is not found
    """
    path = (
        "/Users/haroon/Desktop/personal/FYP/my_Exercise-Correction/web/backend/static"
    )
    path = f"{path}/{file_name}"
    print(path)

    return path if os.path.exists(path) else None


dict_features = {}
left_features = {
    "shoulder": 11,
    "elbow": 13,
    "wrist": 15,
    "hip": 23,
    "knee": 25,
    "ankle": 27,
    "foot": 31,
}

right_features = {
    "shoulder": 12,
    "elbow": 14,
    "wrist": 16,
    "hip": 24,
    "knee": 26,
    "ankle": 28,
    "foot": 32,
}

dict_features["left"] = left_features
dict_features["right"] = right_features
dict_features["nose"] = 0


def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):

    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width

    # draw filled rectangles
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)

    # draw filled ellipses
    cv2.ellipse(
        img,
        (x1 + w, y1 + w),
        (w, w),
        angle=0,
        startAngle=-90,
        endAngle=-180,
        color=box_color,
        thickness=-1,
    )

    cv2.ellipse(
        img,
        (x2 - w, y1 + w),
        (w, w),
        angle=0,
        startAngle=0,
        endAngle=-90,
        color=box_color,
        thickness=-1,
    )

    cv2.ellipse(
        img,
        (x1 + w, y2 - w),
        (w, w),
        angle=0,
        startAngle=90,
        endAngle=180,
        color=box_color,
        thickness=-1,
    )

    cv2.ellipse(
        img,
        (x2 - w, y2 - w),
        (w, w),
        angle=0,
        startAngle=0,
        endAngle=90,
        color=box_color,
        thickness=-1,
    )

    return img


def draw_dotted_line(frame, lm_coord, start, end, line_color):
    pix_step = 0

    for i in range(start, end + 1, 8):
        cv2.circle(
            frame, (lm_coord[0], i + pix_step), 2, line_color, -1, lineType=cv2.LINE_AA
        )

    return frame


def draw_text(
    img,
    msg,
    width=8,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    pos=(0, 0),
    font_scale=1,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
    box_offset=(20, 10),
):

    offset = box_offset
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(
        m + n - o for m, n, o in zip((x + text_w, y + text_h), offset, (25, 0))
    )

    img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)

    cv2.putText(
        img,
        msg,
        (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return text_size


def get_landmark_array_2(
    pose_landmark, key, frame_width, frame_height, VISIBILITY_THRESHOLD=0.5
):
    landmark = pose_landmark[key]
    if landmark.visibility < VISIBILITY_THRESHOLD:
        # Landmark not visible enough, return None
        return None

    denorm_x = int(landmark.x * frame_width)
    denorm_y = int(landmark.y * frame_height)

    visibility = landmark.visibility

    return np.array([denorm_x, denorm_y, landmark.z, visibility])


def get_landmark_array(pose_landmark, key, VISIBILITY_THRESHOLD=0.5):

    if pose_landmark[key]["visibility"] < VISIBILITY_THRESHOLD:
        # print(f"Landmark {key} not visible enough (Visibility: {pose_landmark[key]["visibility"]})")
        return None

    denorm_x = int(pose_landmark[key]["x"])
    denorm_y = int(pose_landmark[key]["y"])
    denorm_z = int(pose_landmark[key]["z"])

    visibility = pose_landmark[key]["visibility"]

    return np.array([denorm_x, denorm_y, denorm_z, visibility])


def get_landmark_features(
    kp_results, dict_features, feature, frame_width, frame_height
):

    if feature == "nose":
        return get_landmark_array_2(
            kp_results, dict_features[feature], frame_width, frame_height
        )

    elif feature == "left" or "right":
        shldr_coord = get_landmark_array_2(
            kp_results, dict_features[feature]["shoulder"], frame_width, frame_height
        )
        elbow_coord = get_landmark_array_2(
            kp_results, dict_features[feature]["elbow"], frame_width, frame_height
        )
        wrist_coord = get_landmark_array_2(
            kp_results, dict_features[feature]["wrist"], frame_width, frame_height
        )
        hip_coord = get_landmark_array_2(
            kp_results, dict_features[feature]["hip"], frame_width, frame_height
        )
        knee_coord = get_landmark_array_2(
            kp_results, dict_features[feature]["knee"], frame_width, frame_height
        )
        ankle_coord = get_landmark_array_2(
            kp_results, dict_features[feature]["ankle"], frame_width, frame_height
        )
        foot_coord = get_landmark_array_2(
            kp_results, dict_features[feature]["foot"], frame_width, frame_height
        )

        return (
            shldr_coord,
            elbow_coord,
            wrist_coord,
            hip_coord,
            knee_coord,
            ankle_coord,
            foot_coord,
        )

    else:
        raise ValueError("feature needs to be either 'nose', 'left' or 'right")


def get_mediapipe_pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    pose = mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return pose


def safe_convert_to_int(degree):
    if math.isnan(degree):
        # Handle NaN case, maybe return a default value like 0 or None
        return None  # or return a default value like 0
    else:
        return int(degree)


def find_angle_3d(p1, p2, p3):
    """
    Calculate the angle between three 3D points.
    p1, p2, and p3 should be numpy arrays with shape (3,) containing x, y, z coordinates.
    p2 is the vertex of the angle (in this case, the elbow).
    """
    # Convert points to numpy arrays if they aren't already

    try:
        if p1 is None or p2 is None or p3 is None:
            return None
        p1 = p1[:3]
        p2 = p2[:3]
        p3 = p3[:3]

        print(p1, p2, p3)
        p1, p2, p3 = map(np.array, (p1, p2, p3))

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate dot product
        dot_product = np.dot(v1, v2)

        # Calculate magnitudes
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        # Calculate angle
        cos_angle = dot_product / (v1_mag * v2_mag)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)

        return int(angle_deg)
    except Exception as e:
        print(e)
        return None


def find_angle(p1, p2, ref_pt=np.array([0, 0])):
    # make points 2d from 3d removing z axis

    if p1 is None or p2 is None or ref_pt is None:
        return None
    p1 = p1[:2]
    p2 = p2[:2]
    ref_pt = ref_pt[:2]
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref, p2_ref)) / (
        1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref)
    )
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    degree = int(180 / np.pi) * theta

    return safe_convert_to_int(degree)


def get_coords_to_use(
    left_shldr_coord,
    left_elbow_coord,
    left_wrist_coord,
    left_hip_coord,
    left_knee_coord,
    left_ankle_coord,
    left_foot_coord,
    right_shldr_coord,
    right_elbow_coord,
    right_wrist_coord,
    right_hip_coord,
    right_knee_coord,
    right_ankle_coord,
    right_foot_coord,
):

    left_shldr_coord_visibility = (
        left_shldr_coord[3] if left_shldr_coord is not None else 0
    )
    left_elbow_coord_visibility = (
        left_elbow_coord[3] if left_elbow_coord is not None else 0
    )
    left_wrist_coord_visibility = (
        left_wrist_coord[3] if left_wrist_coord is not None else 0
    )
    left_hip_coord_visibility = left_hip_coord[3] if left_hip_coord is not None else 0
    left_knee_coord_visibility = (
        left_knee_coord[3] if left_knee_coord is not None else 0
    )
    left_ankle_coord_visibility = (
        left_ankle_coord[3] if left_ankle_coord is not None else 0
    )
    left_foot_coord_visibility = (
        left_foot_coord[3] if left_foot_coord is not None else 0
    )

    right_shldr_coord_visibility = (
        right_shldr_coord[3] if right_shldr_coord is not None else 0
    )
    right_elbow_coord_visibility = (
        right_elbow_coord[3] if right_elbow_coord is not None else 0
    )
    right_wrist_coord_visibility = (
        right_wrist_coord[3] if right_wrist_coord is not None else 0
    )
    right_hip_coord_visibility = (
        right_hip_coord[3] if right_hip_coord is not None else 0
    )
    right_knee_coord_visibility = (
        right_knee_coord[3] if right_knee_coord is not None else 0
    )
    right_ankle_coord_visibility = (
        right_ankle_coord[3] if right_ankle_coord is not None else 0
    )
    right_foot_coord_visibility = (
        right_foot_coord[3] if right_foot_coord is not None else 0
    )

    sum_left = (
        left_shldr_coord_visibility
        + left_elbow_coord_visibility
        + left_wrist_coord_visibility
        + left_hip_coord_visibility
        + left_knee_coord_visibility
        + left_ankle_coord_visibility
        + left_foot_coord_visibility
    )
    sum_right = (
        right_shldr_coord_visibility
        + right_elbow_coord_visibility
        + right_wrist_coord_visibility
        + right_hip_coord_visibility
        + right_knee_coord_visibility
        + right_ankle_coord_visibility
        + right_foot_coord_visibility
    )

    if sum_left > sum_right:
        shldr_coord = left_shldr_coord
        elbow_coord = left_elbow_coord
        wrist_coord = left_wrist_coord
        hip_coord = left_hip_coord
        knee_coord = left_knee_coord
        ankle_coord = left_ankle_coord
        foot_coord = left_foot_coord
        multiplier = -1
    else:
        shldr_coord = right_shldr_coord
        elbow_coord = right_elbow_coord
        wrist_coord = right_wrist_coord
        hip_coord = right_hip_coord
        knee_coord = right_knee_coord
        ankle_coord = right_ankle_coord
        foot_coord = right_foot_coord
        multiplier = 1

    return (
        shldr_coord,
        elbow_coord,
        wrist_coord,
        hip_coord,
        knee_coord,
        ankle_coord,
        foot_coord,
        multiplier,
    )


def get_complete_coords(lm, frame_width, frame_height):
    (
        left_shldr_coord,
        left_elbow_coord,
        left_wrist_coord,
        left_hip_coord,
        left_knee_coord,
        left_ankle_coord,
        left_foot_coord,
    ) = get_landmark_features(lm, dict_features, "left", frame_width, frame_height)
    (
        right_shldr_coord,
        right_elbow_coord,
        right_wrist_coord,
        right_hip_coord,
        right_knee_coord,
        right_ankle_coord,
        right_foot_coord,
    ) = get_landmark_features(lm, dict_features, "right", frame_width, frame_height)
    nose_coord = get_landmark_features(
        lm, dict_features, "nose", frame_width, frame_height
    )

    (
        shldr_coord,
        elbow_coord,
        wrist_coord,
        hip_coord,
        knee_coord,
        ankle_coord,
        foot_coord,
        dir,
    ) = get_coords_to_use(
        left_shldr_coord,
        left_elbow_coord,
        left_wrist_coord,
        left_hip_coord,
        left_knee_coord,
        left_ankle_coord,
        left_foot_coord,
        right_shldr_coord,
        right_elbow_coord,
        right_wrist_coord,
        right_hip_coord,
        right_knee_coord,
        right_ankle_coord,
        right_foot_coord,
    )
    return {
        "left_shldr_coord": left_shldr_coord,
        "left_elbow_coord": left_elbow_coord,
        "left_wrist_coord": left_wrist_coord,
        "left_hip_coord": left_hip_coord,
        "left_knee_coord": left_knee_coord,
        "left_ankle_coord": left_ankle_coord,
        "left_foot_coord": left_foot_coord,
        "right_shldr_coord": right_shldr_coord,
        "right_elbow_coord": right_elbow_coord,
        "right_wrist_coord": right_wrist_coord,
        "right_hip_coord": right_hip_coord,
        "right_knee_coord": right_knee_coord,
        "right_ankle_coord": right_ankle_coord,
        "right_foot_coord": right_foot_coord,
        "nose_coord": nose_coord,
        "shldr_coord": shldr_coord,
        "elbow_coord": elbow_coord,
        "wrist_coord": wrist_coord,
        "hip_coord": hip_coord,
        "knee_coord": knee_coord,
        "ankle_coord": ankle_coord,
        "foot_coord": foot_coord,
        "multiplier": dir,
    }


colors = {
    "blue": (0, 127, 255),
    "red": (255, 50, 50),
    "green": (0, 255, 127),
    "light_green": (100, 233, 127),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "cyan": (0, 255, 255),
    "light_blue": (102, 204, 255),
}


def normalize_sequences(sequences):
    arr_sequences = np.array(sequences)

    # X_normalized = []
    # model = None
    # if normalize:
    scaler = MinMaxScaler()
    num_samples, sequence_length, num_features = arr_sequences.shape
    arr_reshaped = arr_sequences.reshape(-1, num_features)
    # model = load_model("model.keras")
    # Fit and transform
    X_normalized = scaler.fit_transform(arr_reshaped)
    X_normalized = X_normalized.reshape(num_samples, sequence_length, num_features)
    return X_normalized
