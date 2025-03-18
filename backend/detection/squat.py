import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from state_tracker import ExerciseStateTracker
from utils import (
    calculate_distance,
    extract_important_keypoints,
    get_static_file_url,
    get_drawing_color,
    get_complete_coords,
    calculate_angle,
    find_angle,
)


import builtins

# Save the original print function
original_print = print


def print(*args, **kwargs):
    # Open file in append mode and write the printed message
    with open("output.txt", "a") as f:
        # Ensure that all the arguments are converted to string and joined with spaces
        message = " ".join(str(arg) for arg in args)
        # Write the message plus a newline to the file
        f.write(message + "\n")
    # Optionally also call the original print to output to the console
    original_print(*args, **kwargs)


def get_squat_state(angles, thresholds):
    knee = None
    knee_angle = int(angles["knee_vertical_angle_2"])
    print(f"Knee angle: {knee_angle}")
    if (
        thresholds["HIP_KNEE_VERT"]["NORMAL"][0]
        <= knee_angle
        <= thresholds["HIP_KNEE_VERT"]["NORMAL"][1]
    ):
        knee = 1

    elif thresholds["HIP_KNEE_VERT"]["PASS"][0] <= knee_angle:
        knee = 0

    return knee


def calc_squat_coords(lm, frame_width, frame_height):
    coords = get_complete_coords(lm, frame_width, frame_height)
    if (
        coords["shldr_coord"] is None
        or coords["elbow_coord"] is None
        or coords["wrist_coord"] is None
        or coords["hip_coord"] is None
        or coords["knee_coord"] is None
        or coords["ankle_coord"] is None
        or coords["foot_coord"] is None
    ):
        return False
    return coords


def calc_squat_angles(coords):
    print(f"calculating angles from Coords: {coords}")
    offset_angle = calculate_angle(
        coords["left_shldr_coord"], coords["right_shldr_coord"], coords["nose_coord"]
    )
    hip_vertical_angle = calculate_angle(
        coords["shldr_coord"],
        np.array([coords["hip_coord"][0], 0]),
        coords["hip_coord"],
    )
    knee_vertical_angle = calculate_angle(
        coords["hip_coord"],
        np.array([coords["knee_coord"][0], 0]),
        coords["knee_coord"],
    )
    ankle_vertical_angle = calculate_angle(
        coords["knee_coord"],
        np.array([coords["ankle_coord"][0], 0]),
        coords["ankle_coord"],
    )

    offset_angle_2 = find_angle(
        coords["left_shldr_coord"], coords["right_shldr_coord"], coords["nose_coord"]
    )
    hip_vertical_angle_2 = find_angle(
        coords["shldr_coord"],
        np.array([coords["hip_coord"][0], 0]),
        coords["hip_coord"],
    )
    knee_vertical_angle_2 = find_angle(
        coords["hip_coord"],
        np.array([coords["knee_coord"][0], 0]),
        coords["knee_coord"],
    )
    ankle_vertical_angle_2 = find_angle(
        coords["knee_coord"],
        np.array([coords["ankle_coord"][0], 0]),
        coords["ankle_coord"],
    )

    return {
        "offset_angle": offset_angle,
        "hip_vertical_angle": hip_vertical_angle,
        "knee_vertical_angle": knee_vertical_angle,
        "ankle_vertical_angle": ankle_vertical_angle,
        "offset_angle_2": offset_angle_2,
        "hip_vertical_angle_2": hip_vertical_angle_2,
        "knee_vertical_angle_2": knee_vertical_angle_2,
        "ankle_vertical_angle_2": ankle_vertical_angle_2,
    }


def get_thresholds_beginner():

    _ANGLE_HIP_KNEE_VERT = {"NORMAL": (0, 32), "TRANS": (35, 65), "PASS": (70, 95)}

    thresholds = {
        "HIP_KNEE_VERT": _ANGLE_HIP_KNEE_VERT,
        "HIP_THRESH": [10, 50],
        "ANKLE_THRESH": 45,
        "KNEE_THRESH": [50, 70, 95],
        "OFFSET_THRESH": 35.0,
        "INACTIVE_THRESH": 15.0,
        "CNT_FRAME_THRESH": 50,
    }

    return thresholds


def analyze_angles(
    angles,
    stage: str,
    thresholds: dict,
):
    analyzed_results = {
        "squat_too_deep": False,
        "knee_over_toe": False,
        "hip_too_high": False,
    }

    if stage == "down":
        if angles["knee_vertical_angle"] > thresholds["KNEE_THRESH"][2]:
            analyzed_results["squat_too_deep"] = True
        if angles["ankle_vertical_angle"] > thresholds["ANKLE_THRESH"]:
            analyzed_results["knee_over_toe"] = True
        if (
            thresholds["KNEE_THRESH"][0]
            < angles["knee_vertical_angle"]
            < thresholds["KNEE_THRESH"][1]
        ):
            analyzed_results["hip_too_high"] = True

    return analyzed_results

    #     elif angles["knee_vertical_angle"] > thresholds["KNEE_THRESH"][2]:
    #         state_tracker["DISPLAY_TEXT"][3] = True
    #         state_tracker["INCORRECT_POSTURE"] = True
    #         print("squat too deep", state_tracker["state_seq"])

    #     if angles["ankle_vertical_angle"] > thresholds["ANKLE_THRESH"]:
    #         state_tracker["DISPLAY_TEXT"][2] = True
    #         state_tracker["INCORRECT_POSTURE"] = True
    #         print("knee falling over toe", state_tracker["state_seq"])


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def analyze_foot_knee_placement(
    results,
    stage: str,
    foot_shoulder_ratio_thresholds: list,
    knee_foot_ratio_thresholds: dict,
    visibility_threshold: int,
) -> dict:
    """
    Calculate the ratio between the foot and shoulder for FOOT PLACEMENT analysis

    Calculate the ratio between the knee and foot for KNEE PLACEMENT analysis

    Return result explanation:
        -1: Unknown result due to poor visibility
        0: Correct knee placement
        1: Placement too tight
        2: Placement too wide
    """
    analyzed_results = {
        "foot_placement": -1,
        "knee_placement": -1,
    }

    landmarks = results.pose_landmarks.landmark

    # * Visibility check of important landmarks for foot placement analysis
    left_foot_index_vis = landmarks[
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    ].visibility
    right_foot_index_vis = landmarks[
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
    ].visibility

    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (
        left_foot_index_vis < visibility_threshold
        or right_foot_index_vis < visibility_threshold
        or left_knee_vis < visibility_threshold
        or right_knee_vis < visibility_threshold
    ):
        return analyzed_results

    # * Calculate shoulder width
    left_shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
    ]
    right_shoulder = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
    ]
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)

    # * Calculate 2-foot width
    left_foot_index = [
        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
    ]
    right_foot_index = [
        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
    ]
    foot_width = calculate_distance(left_foot_index, right_foot_index)

    # * Calculate foot and shoulder ratio
    foot_shoulder_ratio = round(foot_width / shoulder_width, 1)

    # * Analyze FOOT PLACEMENT
    min_ratio_foot_shoulder, max_ratio_foot_shoulder = foot_shoulder_ratio_thresholds
    if min_ratio_foot_shoulder <= foot_shoulder_ratio <= max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 0
    elif foot_shoulder_ratio < min_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 1
    elif foot_shoulder_ratio > max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 2

    # * Visibility check of important landmarks for knee placement analysis
    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold:
        print("Cannot see foot")
        return analyzed_results

    # * Calculate 2 knee width
    left_knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
    ]
    right_knee = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
    ]
    knee_width = calculate_distance(left_knee, right_knee)

    # * Calculate foot and shoulder ratio
    knee_foot_ratio = round(knee_width / foot_width, 1)

    # * Analyze KNEE placement
    up_min_ratio_knee_foot, up_max_ratio_knee_foot = knee_foot_ratio_thresholds.get(
        "up"
    )
    (
        middle_min_ratio_knee_foot,
        middle_max_ratio_knee_foot,
    ) = knee_foot_ratio_thresholds.get("middle")
    down_min_ratio_knee_foot, down_max_ratio_knee_foot = knee_foot_ratio_thresholds.get(
        "down"
    )

    if stage == "up":
        if up_min_ratio_knee_foot <= knee_foot_ratio <= up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < up_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    elif stage == "middle":
        if middle_min_ratio_knee_foot <= knee_foot_ratio <= middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < middle_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    elif stage == "down":
        if down_min_ratio_knee_foot <= knee_foot_ratio <= down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < down_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2

    return analyzed_results


class SquatDetection:
    ML_MODEL_PATH = get_static_file_url("model/LR_model_new.pkl")

    PREDICTION_PROB_THRESHOLD = 0.63
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
    KNEE_FOOT_RATIO_THRESHOLDS = {
        "up": [0.5, 1.0],
        "middle": [0.7, 1.0],
        "down": [0.7, 1.1],
    }

    def __init__(self) -> None:
        self.init_important_landmarks()
        self.load_machine_learning_model()

        self.current_stage = ""
        self.previous_stage = {
            "feet": "",
            "knee": "",
        }
        self.counter = 0
        self.results = []
        self.has_error = False
        self.state = ExerciseStateTracker()

    def init_important_landmarks(self) -> None:
        """
        Determine Important landmarks for squat detection
        """

        self.important_landmarks = [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
        ]

        # Generate all columns of the data frame
        self.headers = ["label"]  # Label column

        for lm in self.important_landmarks:
            self.headers += [
                f"{lm.lower()}_x",
                f"{lm.lower()}_y",
                f"{lm.lower()}_z",
                f"{lm.lower()}_v",
            ]

    def load_machine_learning_model(self) -> None:
        """
        Load machine learning model
        """
        if not self.ML_MODEL_PATH:
            raise Exception("Cannot found squat model")

        try:
            with open(self.ML_MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model, {e}")

    def handle_detected_results(self, video_name: str) -> tuple:
        """
        Save error frame as evidence
        """
        file_name, _ = video_name.split(".")
        save_folder = get_static_file_url("images")
        for index, error in enumerate(self.results):
            try:
                image_name = f"{file_name}_{index}.jpg"
                cv2.imwrite(f"{save_folder}/{file_name}_{index}.jpg", error["frame"])
                self.results[index]["frame"] = image_name
            except Exception as e:
                print("ERROR cannot save frame: " + str(e))
                self.results[index]["frame"] = None

        return self.results, self.counter

    def clear_results(self) -> None:
        self.current_stage = ""
        self.previous_stage = {
            "feet": "",
            "knee": "",
        }
        self.counter = 0
        self.results = []
        self.has_error = False
        self.state.reset()

    def detect(self, mp_results, image, timestamp) -> None:
        """
        Make Squat Errors detection
        """
        try:
            # * Model prediction for SQUAT counter
            # Extract keypoints from frame for the input
            frame_height, frame_width, _ = image.shape
            coords = calc_squat_coords(
                mp_results.pose_landmarks.landmark, frame_width, frame_height
            )
            if not coords:
                cv2.putText(
                    image,
                    "Cannot detect keypoints",
                    (10, 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.3,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                return
            angles = calc_squat_angles(coords)
            print(f"Angles: {angles}")
            state = get_squat_state(angles, get_thresholds_beginner())
            print(f"State from angles: {state}")
            row = extract_important_keypoints(mp_results, self.important_landmarks)
            X = pd.DataFrame([row], columns=self.headers[1:])

            # Make prediction and its probability
            predicted_class = self.model.predict(X)[0]
            if predicted_class == 0:
                predicted_class = "down"
            else:
                predicted_class = "up"

            prediction_probabilities = self.model.predict_proba(X)[0]
            prediction_probability = round(
                prediction_probabilities[prediction_probabilities.argmax()], 2
            )

            print(f"Predicted class: {predicted_class} with {prediction_probability}")
            # Evaluate model prediction
            if (
                predicted_class == "down"
                and (prediction_probability >= self.PREDICTION_PROB_THRESHOLD - 0.08)
            ) or state == 0:
                self.state.curr_state = "down"
                print(self.state.curr_state, "after setting self.state")
                self.current_stage = "down"
            elif (
                # self.state.curr_state == "down"
                self.current_stage == "down"
                and predicted_class == "up"
                and prediction_probability >= self.PREDICTION_PROB_THRESHOLD
            ):
                self.current_stage = "up"
                self.state.curr_state = "up"
                self.state.REP_COUNT += 1
                self.counter += 1

            print(f"current stage: {self.current_stage}")

            analyzed_angles = analyze_angles(
                angles,
                stage=self.state.curr_state,
                thresholds=get_thresholds_beginner(),
                state_tracker=self.state,
            )

            # Analyze squat pose
            analyzed_results = analyze_foot_knee_placement(
                results=mp_results,
                stage=self.state.curr_state,
                foot_shoulder_ratio_thresholds=self.FOOT_SHOULDER_RATIO_THRESHOLDS,
                knee_foot_ratio_thresholds=self.KNEE_FOOT_RATIO_THRESHOLDS,
                visibility_threshold=self.VISIBILITY_THRESHOLD,
            )

            foot_placement_evaluation = analyzed_results["foot_placement"]
            knee_placement_evaluation = analyzed_results["knee_placement"]

            # * Evaluate FEET PLACEMENT error
            if foot_placement_evaluation == -1:
                feet_placement = "unknown"
            elif foot_placement_evaluation == 0:
                feet_placement = "correct"
            elif foot_placement_evaluation == 1:
                feet_placement = "too tight"
            elif foot_placement_evaluation == 2:
                feet_placement = "too wide"

            # * Evaluate KNEE PLACEMENT error
            if feet_placement == "correct":
                if knee_placement_evaluation == -1:
                    knee_placement = "unknown"
                elif knee_placement_evaluation == 0:
                    knee_placement = "correct"
                elif knee_placement_evaluation == 1:
                    knee_placement = "too tight"
                elif knee_placement_evaluation == 2:
                    knee_placement = "too wide"
            else:
                knee_placement = "unknown"

            # Stage management for saving results
            # * Feet placement
            if feet_placement in ["too tight", "too wide"]:
                # Stage not change
                if self.previous_stage["feet"] == feet_placement:
                    pass
                # Stage from correct to error
                elif self.previous_stage["feet"] != feet_placement:
                    self.results.append(
                        {
                            "stage": f"feet {feet_placement}",
                            "frame": image,
                            "timestamp": timestamp,
                        }
                    )

                self.previous_stage["feet"] = feet_placement

            # * Knee placement
            if knee_placement in ["too tight", "too wide"]:
                # Stage not change
                if self.previous_stage["knee"] == knee_placement:
                    pass
                # Stage from correct to error
                elif self.previous_stage["knee"] != knee_placement:
                    self.results.append(
                        {
                            "stage": f"knee {knee_placement}",
                            "frame": image,
                            "timestamp": timestamp,
                        }
                    )

                self.previous_stage["knee"] = knee_placement

            if feet_placement in ["too tight", "too wide"] or knee_placement in [
                "too tight",
                "too wide",
            ]:
                self.has_error = True
            else:
                self.has_error = False

            # Visualization
            # Draw landmarks and connections
            landmark_color, connection_color = get_drawing_color(self.has_error)
            mp_drawing.draw_landmarks(
                image,
                mp_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=landmark_color, thickness=2, circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=connection_color, thickness=2, circle_radius=1
                ),
            )

            x_fixed = 400
            y_fixed = 200
            rect_width = 300
            rect_height = 40

            # Draw rectangle at fixed position
            cv2.rectangle(
                image,
                (x_fixed, y_fixed),
                (x_fixed + rect_width, y_fixed + rect_height),
                (245, 117, 16),
                -1,
            )

            # Adjust text positions relative to the fixed position
            cv2.putText(
                image,
                "COUNT",
                (x_fixed + 10, y_fixed + 12),
                cv2.FONT_HERSHEY_COMPLEX,
                0.3,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f'{str(self.state.REP_COUNT)}, {str(self.counter)} ,{str(self.current_stage)}, {predicted_class.split(" ")[0]}, {str(prediction_probability)}',
                (x_fixed + 5, y_fixed + 25),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Adjust Feet label position
            cv2.putText(
                image,
                "FEET",
                (x_fixed + 160, y_fixed + 12),
                cv2.FONT_HERSHEY_COMPLEX,
                0.3,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                feet_placement,
                (x_fixed + 155, y_fixed + 25),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Adjust Knee label position
            cv2.putText(
                image,
                "KNEE",
                (x_fixed + 225, y_fixed + 12),
                cv2.FONT_HERSHEY_COMPLEX,
                0.3,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                knee_placement,
                (x_fixed + 220, y_fixed + 25),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            x_rect = image.shape[1] - 250  # Position near the right edge
            y_rect = 50
            rect_width = 230
            rect_height = 180

            # Draw a semi-transparent rectangle
            overlay = image.copy()
            cv2.rectangle(
                overlay,
                (x_rect, y_rect),
                (x_rect + rect_width, y_rect + rect_height),
                (0, 0, 0),
                -1,
            )
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Display text inside the rectangle
            y_offset = y_rect + 20
            text_spacing = 25
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(
                image,
                "Squat Angles",
                (x_rect + 10, y_offset),
                font,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += text_spacing

            for key, value in angles.items():
                cv2.putText(
                    image,
                    f"{key}: {str(value)}°",
                    (x_rect + 10, y_offset),
                    font,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y_offset += text_spacing

            # Status box
            # cv2.rectangle(image, (0, 0), (300, 40), (245, 117, 16), -1)

            # # Display class
            # cv2.putText(
            #     image,
            #     "COUNT",
            #     (10, 12),
            #     cv2.FONT_HERSHEY_COMPLEX,
            #     0.3,
            #     (0, 0, 0),
            #     1,
            #     cv2.LINE_AA,
            # )
            # cv2.putText(
            #     image,
            #     f'{str(self.state.REP_COUNT)}, {predicted_class.split(" ")[0]}, {str(prediction_probability)}',
            #     (5, 25),
            #     cv2.FONT_HERSHEY_COMPLEX,
            #     0.5,
            #     (255, 255, 255),
            #     1,
            #     cv2.LINE_AA,
            # )

            # # Display Feet and Shoulder width ratio
            # cv2.putText(
            #     image,
            #     "FEET",
            #     (130, 12),
            #     cv2.FONT_HERSHEY_COMPLEX,
            #     0.3,
            #     (0, 0, 0),
            #     1,
            #     cv2.LINE_AA,
            # )
            # cv2.putText(
            #     image,
            #     feet_placement,
            #     (125, 25),
            #     cv2.FONT_HERSHEY_COMPLEX,
            #     0.5,
            #     (255, 255, 255),
            #     1,
            #     cv2.LINE_AA,
            # )

            # # Display knee and Shoulder width ratio
            # cv2.putText(
            #     image,
            #     "KNEE",
            #     (225, 12),
            #     cv2.FONT_HERSHEY_COMPLEX,
            #     0.3,
            #     (0, 0, 0),
            #     1,
            #     cv2.LINE_AA,
            # )
            # cv2.putText(
            #     image,
            #     knee_placement,
            #     (220, 25),
            #     cv2.FONT_HERSHEY_COMPLEX,
            #     0.5,
            #     (255, 255, 255),
            #     1,
            #     cv2.LINE_AA,
            # )

        except Exception as e:
            print(f"Error while detecting squat errors: {e}")


# import cv2
# import numpy as np
# import pandas as pd
# import pickle
# from utils import (
#     calculate_distance,
#     extract_important_keypoints,
#     get_static_file_url,
#     get_complete_coords,
#     find_angle,
#     draw_text,
#     draw_dotted_line,
# )
# import mediapipe as mp

# mp_pose = mp.solutions.pose


# class ComprehensiveSquatTracker:
#     ML_MODEL_PATH = get_static_file_url("model/LR_model_new.pkl")
#     PREDICTION_PROB_THRESHOLD = 0.6
#     VISIBILITY_THRESHOLD = 0.6
#     FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
#     KNEE_FOOT_RATIO_THRESHOLDS = {
#         "up": [0.5, 1.0],
#         "middle": [0.7, 1.0],
#         "down": [0.7, 1.1],
#     }

#     def __init__(self):
#         self._load_ml_model()
#         self._init_landmarks()

#         self.current_stage = ""
#         self.previous_stage = ""
#         self.counter = 0
#         self.errors = []
#         self.has_error = False

#     def _load_ml_model(self):
#         with open(self.ML_MODEL_PATH, "rb") as f:
#             self.model, self.input_scaler = pickle.load(f)

#     def _init_landmarks(self):
#         self.important_landmarks = [
#             "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP",
#             "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
#         ]
#         self.headers = ["label"] + [
#             f"{lm.lower()}_{axis}" for lm in self.important_landmarks for axis in ["x", "y", "z", "v"]
#         ]

#     def process_frame(self, mp_results, frame, timestamp):
#         self._predict_stage(mp_results)

#         # Real-time feedback on angles
#         self._analyze_angles(mp_results, frame)

#         if self.current_stage != self.previous_stage:
#             self._on_stage_change(mp_results, frame, timestamp)

#         self.previous_stage = self.current_stage

#     def _predict_stage(self, mp_results):
#         row = extract_important_keypoints(mp_results, self.important_landmarks)
#         X = pd.DataFrame([row], columns=self.headers[1:])
#         X = pd.DataFrame(self.input_scaler.transform(X))

#         predicted_class = self.model.predict(X)[0]
#         prediction_probabilities = self.model.predict_proba(X)[0]
#         probability = prediction_probabilities.max()

#         if probability < self.PREDICTION_PROB_THRESHOLD:
#             return

#         self.current_stage = "down" if predicted_class == 0 else "up"

#     def _on_stage_change(self, mp_results, frame, timestamp):
#         if self.current_stage == "down":
#             self._check_foot_knee_placement(mp_results, frame, timestamp)

#         elif self.current_stage == "up" and self.previous_stage == "down":
#             self.counter += 1  # Only count rep when moving up from down

#     def _check_foot_knee_placement(self, mp_results, frame, timestamp):
#         analyzed_results = self._analyze_foot_knee_placement(mp_results)
#         if analyzed_results["foot_placement"] not in [0, -1]:
#             self.errors.append({
#                 "stage": f"Feet {self._placement_message(analyzed_results['foot_placement'])}",
#                 "frame": frame.copy(),
#                 "timestamp": timestamp
#             })

#         if analyzed_results["knee_placement"] not in [0, -1]:
#             self.errors.append({
#                 "stage": f"Knee {self._placement_message(analyzed_results['knee_placement'])}",
#                 "frame": frame.copy(),
#                 "timestamp": timestamp
#             })

#     def _analyze_foot_knee_placement(self, results):
#         landmarks = results.pose_landmarks.landmark

#         left_foot = np.array([landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
#                               landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y])
#         right_foot = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
#                                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y])
#         left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
#                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
#         right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
#                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
#         foot_width = calculate_distance(left_foot, right_foot)
#         shoulder_width = calculate_distance(left_shoulder, right_shoulder)

#         foot_shoulder_ratio = foot_width / shoulder_width
#         foot_placement = self._evaluate_ratio(foot_shoulder_ratio, self.FOOT_SHOULDER_RATIO_THRESHOLDS)

#         left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
#                               landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y])
#         right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
#                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y])
#         knee_width = calculate_distance(left_knee, right_knee)

#         knee_foot_ratio = knee_width / foot_width
#         knee_placement = self._evaluate_ratio(knee_foot_ratio, self.KNEE_FOOT_RATIO_THRESHOLDS['down'])

#         return {"foot_placement": foot_placement, "knee_placement": knee_placement}

#     def _evaluate_ratio(self, ratio, thresholds):
#         min_ratio, max_ratio = thresholds
#         if min_ratio <= ratio <= max_ratio:
#             return 0  # Correct
#         elif ratio < min_ratio:
#             return 1  # Too tight
#         elif ratio > max_ratio:
#             return 2  # Too wide
#         return -1  # Unknown

#     def _placement_message(self, code):
#         return {1: "Too Tight", 2: "Too Wide"}.get(code, "Unknown")

#     def _analyze_angles(self, mp_results, frame):
#         coords = get_complete_coords(mp_results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])
#         if not coords:
#             return

#         hip_angle = find_angle(coords["shldr_coord"], coords["hip_coord"], coords["knee_coord"])
#         knee_angle = find_angle(coords["hip_coord"], coords["knee_coord"], coords["ankle_coord"])

#         if hip_angle > 50:
#             draw_text(frame, "Lean Backwards", (50, 50), (0, 0, 255))
#         elif hip_angle < 10:
#             draw_text(frame, "Bend Forward", (50, 80), (0, 0, 255))

#         if knee_angle > 95:
#             draw_text(frame, "Squat Too Deep", (50, 110), (0, 0, 255))

#     def get_summary(self):
#         return {
#             "reps": self.counter,
#             "errors": self.errors
#         }
