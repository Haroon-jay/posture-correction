import mediapipe as mp
import cv2
import numpy as np
from utils import find_angle, find_angle_3d, draw_text, get_complete_coords
from thresholds import get_thresholds_curl

from configs.exercise_config import ExerciseConfig


def get_curl_state(angles, thresholds, isFront=True):
    elbow = None
    elbow_angle = int(angles["elbow_angle"])
    key = "FRONT" if isFront else "SIDE"
    print(elbow_angle, key)
    if (
        thresholds["SHOULDER_ELBOW"][key]["START"][0]
        <= elbow_angle
        <= thresholds["SHOULDER_ELBOW"][key]["START"][1]
    ):
        elbow = 1
    elif thresholds["SHOULDER_ELBOW"][key]["COMPLETE"] <= elbow_angle:
        elbow = 2

    return f"s{elbow}" if elbow else None


def perform_action_for_state_curl(current_state, state_tracker, angles, thresholds):
    print(current_state, state_tracker["state_seq"])
    if current_state == "s1" and len(state_tracker["state_seq"]) == 2:
        if (
            len(state_tracker["state_seq"]) == 2
            and not state_tracker["INCORRECT_POSTURE"]
        ):
            state_tracker["REP_COUNT"] += 1
        elif state_tracker["INCORRECT_POSTURE"]:
            state_tracker["IMPROPER_REP_COUNT"] += 1
            state_tracker["INCORRECT_POSTURE"] = False
        state_tracker["state_seq"] = []
        state_tracker["INCORRECT_POSTURE"] = False
    elif abs(angles["back_dist"]) > thresholds["BACK_DIST_THRESH"]:
        state_tracker["INCORRECT_POSTURE"] = True
        state_tracker["DISPLAY_TEXT"][0] = True
        print("back not straight")


def on_front_view_curl(coords, angles, state_tracker, COLORS):
    # draw_text(
    #     frame,
    #     "Camera not alligned properly: ",
    #     pos=(int(frame_width * 0.68), 30),
    #     text_color=(255, 255, 230),
    #     font_scale=0.7,
    #     text_color_bg=(18, 185, 0),
    # )

    # draw_text(
    #     frame,
    #     "INCORRECT: " + str(state_tracker["IMPROPER_REP_COUNT"]),
    #     pos=(int(frame_width * 0.68), 80),
    #     text_color=(255, 255, 230),
    #     font_scale=0.7,
    #     text_color_bg=(221, 0, 0),
    # )
    return False


def calc_curl_coords(lm):
    coords = get_complete_coords(lm)
    if (
        coords["shldr_coord"] is None
        or coords["elbow_coord"] is None
        or coords["wrist_coord"] is None
        or coords["hip_coord"] is None
    ):
        return False

    return coords


def calc_curl_angles(coords):
    offset_angle = find_angle(
        coords["left_shldr_coord"], coords["right_shldr_coord"], coords["nose_coord"]
    )
    elbow_angle = find_angle(
        coords["shldr_coord"], coords["elbow_coord"], coords["wrist_coord"]
    )
    elbow_angle_3d = find_angle_3d(
        coords["shldr_coord"], coords["elbow_coord"], coords["wrist_coord"]
    )
    hip_vertical_angle = find_angle(
        coords["shldr_coord"],
        np.array([coords["hip_coord"][0], 0]),
        coords["hip_coord"],
    )

    back_dist = coords["hip_coord"][0] - coords["shldr_coord"][0]
    return {
        "elbow_angle": elbow_angle,
        "hip_vertical_angle": hip_vertical_angle,
        "offset_angle": offset_angle,
        "elbow_angle_3d": elbow_angle_3d,
        "back_dist": back_dist,
    }


def update_state_sequence_curl(state, state_tracker):
    print(state, state_tracker["state_seq"])
    if state == "s1":
        if len(state_tracker["state_seq"]) == 0:
            state_tracker["state_seq"].append(state)
    elif state == "s2":
        if len(state_tracker["state_seq"]) == 1 and "s1" in state_tracker["state_seq"]:
            print("appending s2")
            state_tracker["state_seq"].append(state)


def on_side_view_curl(coords, angles, COLORS, linetype, font):
    return True


curl_config = ExerciseConfig(
    coords_calc_fn=calc_curl_coords,
    angles_calc_fn=calc_curl_angles,
    thresholds_fn=get_thresholds_curl,
    get_state_fn=get_curl_state,
    perform_action_fn=perform_action_for_state_curl,
    view_fns={"front": on_front_view_curl, "side": on_side_view_curl},
    feedback_map={
        0: ("STRAIGHTEN BACK", 215, (0, 153, 255)),
        1: ("BEND FORWARD", 215, (0, 153, 255)),
    },
    update_state_sequence=update_state_sequence_curl,
)
