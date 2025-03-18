import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import traceback

from utils import (
    calculate_angle,
    extract_important_keypoints,
    get_static_file_url,
    get_drawing_color,
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class PushupDetection:
    VISIBILITY_THRESHOLD = 0.65
    
    # Elbow angle thresholds
    DOWN_ELBOW_ANGLE_THRESHOLD = 90  # When arms are bent
    UP_ELBOW_ANGLE_THRESHOLD = 160  # When arms are straight
    
    # Body alignment threshold (180 is perfect straight line)
    BODY_ALIGNMENT_THRESHOLD = 160
    
    # Wrist-shoulder alignment threshold (degrees)
    WRIST_POSITION_THRESHOLD = 30

    def __init__(self) -> None:
        self.init_important_landmarks()
        
        self.counter = 0
        self.stage = "up"  # Start in up position for pushups
        
        self.results = []
        self.has_error = False
        
        # Error tracking
        self.elbow_angle_error = False
        self.body_alignment_error = False
        self.wrist_position_error = False

    def init_important_landmarks(self) -> None:
        """
        Determine Important landmarks for pushup detection
        """
        self.important_landmarks = [
            "NOSE",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "RIGHT_ELBOW",
            "LEFT_ELBOW",
            "RIGHT_WRIST",
            "LEFT_WRIST",
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

    def handle_detected_results(self, video_name: str) -> tuple:
        """
        Save frame as evidence
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

        return self.results, {
            "counter": self.counter,
        }

    def clear_results(self) -> None:
        self.results = []
        self.has_error = False
        self.counter = 0
        self.stage = "up"
        self.elbow_angle_error = False
        self.body_alignment_error = False
        self.wrist_position_error = False

    def detect(
        self,
        mp_results,
        image,
        timestamp: int,
    ) -> None:
        """Error detection

        Args:
            mp_results (): MediaPipe results
            image (): OpenCV image
            timestamp (int): Current time of the frame
        """
        self.has_error = False

        try:
            video_dimensions = [image.shape[1], image.shape[0]]
            landmarks = mp_results.pose_landmarks.landmark

            # Check visibility
            joints_visibility = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility,
            ]

            is_visible = all([vis > self.VISIBILITY_THRESHOLD for vis in joints_visibility])
            
            if not is_visible:
                # Skip processing if key points aren't visible
                landmark_color, connection_color = get_drawing_color(False)
                mp_drawing.draw_landmarks(
                    image,
                    mp_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1),
                )
                return

            # Get key joint coordinates 
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            ]
            left_elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            right_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            ]
            left_wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]
            right_wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            ]
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]
            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            ]
            left_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            ]
            right_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            ]

            # Calculate midpoints for analysis
            shoulder_coord = np.mean([left_shoulder, right_shoulder], axis=0)
            elbow_coord = np.mean([left_elbow, right_elbow], axis=0)
            wrist_coord = np.mean([left_wrist, right_wrist], axis=0)
            hip_coord = np.mean([left_hip, right_hip], axis=0)
            knee_coord = np.mean([left_knee, right_knee], axis=0)

            # Calculate angles
            # Elbow angle (key for pushup form)
            elbow_angle = int(calculate_angle(shoulder_coord, elbow_coord, wrist_coord))
            
            # Body line angle (should be straight during pushup)
            body_angle = int(calculate_angle(shoulder_coord, hip_coord, knee_coord))
            
            # Calculate vertical alignment between wrist and shoulder
            wrist_shoulder_vertical = [wrist_coord[0], 0]
            wrist_position_angle = int(
                calculate_angle(wrist_coord, wrist_shoulder_vertical, shoulder_coord)
            )

            # Count pushups
            if self.stage == "down" and elbow_angle > self.UP_ELBOW_ANGLE_THRESHOLD:
                self.stage = "up"
                self.counter += 1
            elif self.stage == "up" and elbow_angle < self.DOWN_ELBOW_ANGLE_THRESHOLD:
                self.stage = "down"

            # Error detection
            # 1. Elbow angle error when in DOWN position
            if self.stage == "down" and (elbow_angle > self.DOWN_ELBOW_ANGLE_THRESHOLD + 30):
                self.has_error = True
                cv2.rectangle(image, (350, 0), (600, 40), (245, 117, 16), -1)
                cv2.putText(
                    image,
                    "PUSHUP ERROR",
                    (360, 12),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "INCOMPLETE PUSHUP",
                    (355, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if not self.elbow_angle_error:
                    self.elbow_angle_error = True
                    self.results.append(
                        {"stage": "incomplete pushup", "frame": image, "timestamp": timestamp}
                    )
            else:
                self.elbow_angle_error = False

            # 2. Body alignment error (body should be straight)
            if body_angle > self.BODY_ALIGNMENT_THRESHOLD:
                self.has_error = True
                cv2.rectangle(image, (350, 45), (600, 85), (245, 117, 16), -1)
                cv2.putText(
                    image,
                    "PUSHUP ERROR",
                    (360, 57),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "BODY NOT STRAIGHT",
                    (355, 75),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if not self.body_alignment_error:
                    self.body_alignment_error = True
                    self.results.append(
                        {"stage": "body not straight", "frame": image, "timestamp": timestamp}
                    )
            else:
                self.body_alignment_error = False

            # 3. Wrist position error (wrists should be under shoulders)
            if wrist_position_angle > self.WRIST_POSITION_THRESHOLD:
                self.has_error = True
                cv2.rectangle(image, (350, 90), (600, 130), (245, 117, 16), -1)
                cv2.putText(
                    image,
                    "PUSHUP ERROR",
                    (360, 102),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "INCORRECT HAND POSITION",
                    (355, 120),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if not self.wrist_position_error:
                    self.wrist_position_error = True
                    self.results.append(
                        {"stage": "incorrect hand position", "frame": image, "timestamp": timestamp}
                    )
            else:
                self.wrist_position_error = False

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

            # Status box
            cv2.rectangle(image, (0, 0), (250, 40), (245, 117, 16), -1)

            # Display counter
            cv2.putText(
                image,
                "COUNT",
                (15, 12),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(self.counter),
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Display current stage
            cv2.putText(
                image,
                "STAGE",
                (95, 12),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                self.stage.upper(),
                (90, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Display pushup status
            cv2.putText(
                image,
                "STATUS",
                (175, 12),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "ERROR" if self.has_error else "GOOD",
                (170, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Visualize angles
            cv2.putText(
                image,
                f"Elbow: {elbow_angle}",
                tuple(
                    np.multiply(
                        elbow_coord, video_dimensions
                    ).astype(int)
                ),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            
            cv2.putText(
                image,
                f"Body: {body_angle}",
                tuple(
                    np.multiply(
                        hip_coord, video_dimensions
                    ).astype(int)
                ),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        except Exception as e:
            traceback.print_exc()
            raise e